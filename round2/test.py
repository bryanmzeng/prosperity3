from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math


class Product:
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC1 = "SYNTHETIC1"
    SYNTHETIC2 = "SYNTHETIC2"
    SPREAD = "SPREAD"
    SPREAD2 = "SPREAD2"


PARAMS = {
    Product.SPREAD: {
        "default_spread_mean": 46.7624, #46.7624 at 7.8k, 48.7624 original
        "default_spread_std": 83.5354,
        "spread_std_window": 55, #55 best
        "zscore_threshold": 3, # 3 is at 6.8k
        "target_position": 58, #58 original w/o basket 2
    },
    Product.SPREAD2: {
        "default_spread_mean": 30.2359,
        "default_spread_std": 54.0495,
        "spread_std_window": 55, #55 best
        "zscore_threshold": 3, # 3 is at 6.8k
        "target_position": -80, #-60 12k -> try -55 for more stable
    }
}

BASKET_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}
BASKET2_WEIGHTS = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2,
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
        }



    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_SYNTHETIC1_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CROISSANTS_PER_BASKET = BASKET_WEIGHTS[Product.CROISSANTS]
        JAMS_PER_BASKET = BASKET_WEIGHTS[Product.JAMS]
        DJEMBES_PER_BASKET = BASKET_WEIGHTS[Product.DJEMBES]

        # Initialize the SYNTHETIC1 basket order depth
        SYNTHETIC1_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        CROISSANTS_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        CROISSANTS_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        JAMS_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        JAMS_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        DJEMBES_best_bid = (
            max(order_depths[Product.DJEMBES].buy_orders.keys())
            if order_depths[Product.DJEMBES].buy_orders
            else 0
        )
        DJEMBES_best_ask = (
            min(order_depths[Product.DJEMBES].sell_orders.keys())
            if order_depths[Product.DJEMBES].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the SYNTHETIC1 basket
        implied_bid = (
            CROISSANTS_best_bid * CROISSANTS_PER_BASKET
            + JAMS_best_bid * JAMS_PER_BASKET
            + DJEMBES_best_bid * DJEMBES_PER_BASKET
        )
        implied_ask = (
            CROISSANTS_best_ask * CROISSANTS_PER_BASKET
            + JAMS_best_ask * JAMS_PER_BASKET
            + DJEMBES_best_ask * DJEMBES_PER_BASKET
        )

        # Calculate the maximum number of SYNTHETIC1 baskets available at the implied bid and ask
        if implied_bid > 0:
            CROISSANTS_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[CROISSANTS_best_bid]
                // CROISSANTS_PER_BASKET
            )
            JAMS_bid_volume = (
                order_depths[Product.JAMS].buy_orders[JAMS_best_bid]
                // JAMS_PER_BASKET
            )
            DJEMBES_bid_volume = (
                order_depths[Product.DJEMBES].buy_orders[DJEMBES_best_bid]
                // DJEMBES_PER_BASKET
            )
            implied_bid_volume = min(
                CROISSANTS_bid_volume, JAMS_bid_volume, DJEMBES_bid_volume
            )
            SYNTHETIC1_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            CROISSANTS_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[CROISSANTS_best_ask]
                // CROISSANTS_PER_BASKET
            )
            JAMS_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[JAMS_best_ask]
                // JAMS_PER_BASKET
            )
            DJEMBES_ask_volume = (
                -order_depths[Product.DJEMBES].sell_orders[DJEMBES_best_ask]
                // DJEMBES_PER_BASKET
            )
            implied_ask_volume = min(
                CROISSANTS_ask_volume, JAMS_ask_volume, DJEMBES_ask_volume
            )
            SYNTHETIC1_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return SYNTHETIC1_order_price

    def convert_SYNTHETIC1_basket_orders(
        self, SYNTHETIC1_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: [],
        }

        # Get the best bid and ask for the SYNTHETIC1 basket
        SYNTHETIC1_basket_order_depth = self.get_SYNTHETIC1_basket_order_depth(
            order_depths
        )
        best_bid = (
            max(SYNTHETIC1_basket_order_depth.buy_orders.keys())
            if SYNTHETIC1_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(SYNTHETIC1_basket_order_depth.sell_orders.keys())
            if SYNTHETIC1_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each SYNTHETIC1 basket order
        for order in SYNTHETIC1_orders:
            # Extract the price and quantity from the SYNTHETIC1 basket order
            price = order.price
            quantity = order.quantity

            # Check if the SYNTHETIC1 basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                CROISSANTS_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                JAMS_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                DJEMBES_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                CROISSANTS_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                JAMS_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
                DJEMBES_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                # The SYNTHETIC1 basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            CROISSANTS_order = Order(
                Product.CROISSANTS,
                CROISSANTS_price,
                quantity * BASKET_WEIGHTS[Product.CROISSANTS],
            )
            JAMS_order = Order(
                Product.JAMS,
                JAMS_price,
                quantity * BASKET_WEIGHTS[Product.JAMS],
            )
            DJEMBES_order = Order(
                Product.DJEMBES, DJEMBES_price, quantity * BASKET_WEIGHTS[Product.DJEMBES]
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(CROISSANTS_order)
            component_orders[Product.JAMS].append(JAMS_order)
            component_orders[Product.DJEMBES].append(DJEMBES_order)

        return component_orders
    def get_SYNTHETIC2_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CROISSANTS_PER_BASKET = BASKET2_WEIGHTS[Product.CROISSANTS]
        JAMS_PER_BASKET = BASKET2_WEIGHTS[Product.JAMS]

        # Initialize the SYNTHETIC1 basket order depth
        SYNTHETIC2_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        CROISSANTS_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        CROISSANTS_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        JAMS_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        JAMS_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the SYNTHETIC1 basket
        implied_bid = (
            CROISSANTS_best_bid * CROISSANTS_PER_BASKET
            + JAMS_best_bid * JAMS_PER_BASKET
        )
        implied_ask = (
            CROISSANTS_best_ask * CROISSANTS_PER_BASKET
            + JAMS_best_ask * JAMS_PER_BASKET
        )

        # Calculate the maximum number of SYNTHETIC1 baskets available at the implied bid and ask
        if implied_bid > 0:
            CROISSANTS_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[CROISSANTS_best_bid]
                // CROISSANTS_PER_BASKET
            )
            JAMS_bid_volume = (
                order_depths[Product.JAMS].buy_orders[JAMS_best_bid]
                // JAMS_PER_BASKET
            )
            implied_bid_volume = min(
                CROISSANTS_bid_volume, JAMS_bid_volume
            )
            SYNTHETIC2_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            CROISSANTS_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[CROISSANTS_best_ask]
                // CROISSANTS_PER_BASKET
            )
            JAMS_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[JAMS_best_ask]
                // JAMS_PER_BASKET
            )
            implied_ask_volume = min(
                CROISSANTS_ask_volume, JAMS_ask_volume
            )
            SYNTHETIC2_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return SYNTHETIC2_order_price

    def convert_SYNTHETIC2_basket_orders(
        self, SYNTHETIC2_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
        }

        # Get the best bid and ask for the SYNTHETIC1 basket
        SYNTHETIC2_basket_order_depth = self.get_SYNTHETIC2_basket_order_depth(
            order_depths
        )
        best_bid = (
            max(SYNTHETIC2_basket_order_depth.buy_orders.keys())
            if SYNTHETIC2_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(SYNTHETIC2_basket_order_depth.sell_orders.keys())
            if SYNTHETIC2_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each SYNTHETIC2 basket order
        for order in SYNTHETIC2_orders:
            # Extract the price and quantity from the SYNTHETIC2 basket order
            price = order.price
            quantity = order.quantity

            # Check if the SYNTHETIC2 basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                CROISSANTS_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                JAMS_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                CROISSANTS_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                JAMS_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
            else:
                # The SYNTHETIC2 basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            CROISSANTS_order = Order(
                Product.CROISSANTS,
                CROISSANTS_price,
                quantity * BASKET2_WEIGHTS[Product.CROISSANTS],
            )
            JAMS_order = Order(
                Product.JAMS,
                JAMS_price,
                quantity * BASKET2_WEIGHTS[Product.JAMS],
            )
            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(CROISSANTS_order)
            component_orders[Product.JAMS].append(JAMS_order)
        return component_orders

    def execute_spread_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        SYNTHETIC1_order_depth = self.get_SYNTHETIC1_basket_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            SYNTHETIC1_bid_price = max(SYNTHETIC1_order_depth.buy_orders.keys())
            SYNTHETIC1_bid_volume = abs(
                SYNTHETIC1_order_depth.buy_orders[SYNTHETIC1_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, SYNTHETIC1_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_ask_price, execute_volume)
            ]
            SYNTHETIC1_orders = [
                Order(Product.SYNTHETIC1, SYNTHETIC1_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_SYNTHETIC1_basket_orders(
                SYNTHETIC1_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            SYNTHETIC1_ask_price = min(SYNTHETIC1_order_depth.sell_orders.keys())
            SYNTHETIC1_ask_volume = abs(
                SYNTHETIC1_order_depth.sell_orders[SYNTHETIC1_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, SYNTHETIC1_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_bid_price, -execute_volume)
            ]
            SYNTHETIC1_orders = [
                Order(Product.SYNTHETIC1, SYNTHETIC1_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_SYNTHETIC1_basket_orders(
                SYNTHETIC1_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders

    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if Product.PICNIC_BASKET1 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        SYNTHETIC1_order_depth = self.get_SYNTHETIC1_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        SYNTHETIC1_swmid = self.get_swmid(SYNTHETIC1_order_depth)
        spread = basket_swmid - SYNTHETIC1_swmid
        spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (
            spread - self.params[Product.SPREAD]["default_spread_mean"]
        ) / spread_std

        if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(
                    -self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(
                    self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None
    def spread2_orders(
    self,
    order_depths: Dict[str, OrderDepth],
    product: Product,
    basket_position: int,
    spread_data: Dict[str, Any],
):
        if Product.PICNIC_BASKET2 not in order_depths:
            return None

        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        SYNTHETIC2_order_depth = self.get_SYNTHETIC2_basket_order_depth(order_depths)

        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(SYNTHETIC2_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if len(spread_data["spread_history"]) < self.params[Product.SPREAD2]["spread_std_window"]:
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD2]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])
        zscore = (spread - self.params[Product.SPREAD2]["default_spread_mean"]) / spread_std

        if zscore >= self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(
                    -self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(
                    self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None
    def execute_spread2_orders(
    self,
    target_position: int,
    basket_position: int,
    order_depths: Dict[str, OrderDepth],
):
        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        SYNTHETIC2_order_depth = self.get_SYNTHETIC2_basket_order_depth(order_depths)

        if target_position > basket_position:
            # Buy ETF2, sell synthetic
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(SYNTHETIC2_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(SYNTHETIC2_order_depth.buy_orders[synthetic_bid_price])

            execute_volume = min(basket_ask_volume, synthetic_bid_volume, target_quantity)

            basket_orders = [Order(Product.PICNIC_BASKET2, basket_ask_price, execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC2, synthetic_bid_price, -execute_volume)]

        else:
            # Sell ETF2, buy synthetic
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(SYNTHETIC2_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(SYNTHETIC2_order_depth.sell_orders[synthetic_ask_price])

            execute_volume = min(basket_bid_volume, synthetic_ask_volume, target_quantity)

            basket_orders = [Order(Product.PICNIC_BASKET2, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC2, synthetic_ask_price, execute_volume)]

        aggregate_orders = self.convert_SYNTHETIC2_basket_orders(synthetic_orders, order_depths)
        aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
        return aggregate_orders



    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0


        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket_position = (
            state.position[Product.PICNIC_BASKET1]
            if Product.PICNIC_BASKET1 in state.position
            else 0
        )
        # spread_orders = self.spread_orders(
        #     state.order_depths,
        #     Product.PICNIC_BASKET1,
        #     basket_position,
        #     traderObject[Product.SPREAD],
        # )
        # if spread_orders != None:
        #     result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS]
        #     result[Product.JAMS] = spread_orders[Product.JAMS]
        #     result[Product.DJEMBES] = spread_orders[Product.DJEMBES]
        #     result[Product.PICNIC_BASKET1] = spread_orders[Product.PICNIC_BASKET1]

        if Product.SPREAD2 not in traderObject:
            traderObject[Product.SPREAD2] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket2_position = (
            state.position.get(Product.PICNIC_BASKET2, 0)
        )

        # spread2_orders = self.spread2_orders(
        #     state.order_depths,
        #     Product.PICNIC_BASKET2,
        #     basket2_position,
        #     traderObject[Product.SPREAD2],
        # )

        # if spread2_orders is not None:
        #     result[Product.CROISSANTS] = result.get(Product.CROISSANTS, []) + spread2_orders[Product.CROISSANTS]
        #     result[Product.JAMS] = result.get(Product.JAMS, []) + spread2_orders[Product.JAMS]
        #     result[Product.PICNIC_BASKET2] = spread2_orders[Product.PICNIC_BASKET2]

        arb_candidates = []

        # ETF1 arb
        spread1_orders = self.spread_orders(
            state.order_depths,
            Product.PICNIC_BASKET1,
            basket_position,
            traderObject[Product.SPREAD],
        )
        if spread1_orders:
            arb_candidates.append(('ETF1', spread1_orders, abs(traderObject[Product.SPREAD]["prev_zscore"])))

        # ETF2 arb
        spread2_orders = self.spread2_orders(
            state.order_depths,
            Product.PICNIC_BASKET2,
            basket2_position,
            traderObject[Product.SPREAD2],
        )
        if spread2_orders:
            arb_candidates.append(('ETF2', spread2_orders, abs(traderObject[Product.SPREAD2]["prev_zscore"])))

        # Sort by z-score magnitude (or expected edge)
        arb_candidates.sort(key=lambda x: x[2], reverse=True)

        # Pick top priority only
        if arb_candidates:
            chosen_arb = arb_candidates[0]
            arb_id, arb_orders, _ = chosen_arb
            if arb_id == 'ETF1':
                result[Product.CROISSANTS] = arb_orders[Product.CROISSANTS]
                result[Product.JAMS] = arb_orders[Product.JAMS]
                result[Product.DJEMBES] = arb_orders[Product.DJEMBES]
                result[Product.PICNIC_BASKET1] = arb_orders[Product.PICNIC_BASKET1]
            elif arb_id == 'ETF2':
                result[Product.CROISSANTS] = arb_orders[Product.CROISSANTS]
                result[Product.JAMS] = arb_orders[Product.JAMS]
                result[Product.PICNIC_BASKET2] = arb_orders[Product.PICNIC_BASKET2]
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData