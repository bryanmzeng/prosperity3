from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import collections


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
        "spread_std_window": 58, #55 best
        "zscore_threshold": 3, # 3 is at 6.8k
        "target_position": 55, #58 original w/o basket 2
    },
    Product.SPREAD2: {
        "default_spread_mean": 30.2359,
        "default_spread_std": 54.0495,
        "spread_std_window": 55, #55 best
        "zscore_threshold": 3, # 3 is at 6.8k
        "target_position": -85, #-60 12k -> try -55 for more stable, -85 at 14k with some drop off near 100k timestep
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

        self.position_limit = 50
        self.target_inventory = 0 # Target inventory level (default: neutral)
        
        
        # State variables
        self.price_history = {}
        self.volatility_estimates = {}
        self.last_mid_prices = {}
        self.timestamps = {}
        self.fair_value_estimates = {}
        self.realized_pnl = {}
        self.trades_completed = {}
        
        
        # KELP specific parameters
        self.kelp_dim = 4  # Number of lags for KELP regression
        self.kelp_cache = []
        self.last_mid_prices = {}  # To store last known mid prices


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
    def process_rainforest_resin_simple(self, state: TradingState, result: dict) -> None:
        symbol = "RAINFOREST_RESIN"
        order_depth = state.order_depths[symbol]
        current_position = state.position.get(symbol, 0)
        resin_orders = []
    
        fair_value = 10000
    
        # Sort order books for easier processing
        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
    
        # Extract best prices
        best_sell_price = min(sell_orders.keys()) if sell_orders else fair_value + 5
        best_buy_price = max(buy_orders.keys()) if buy_orders else fair_value - 5
    
        # Current position
        position = current_position
    
        # STEP 1: Take advantage of any mispriced orders
        # Buy any sell orders below fair value or at fair value if we're short
        for price, volume in sell_orders.items():
            if (price < fair_value or (current_position < 0 and price == fair_value)) and position < self.position_limit:
                order_size = min(-volume, self.position_limit - position)
                if order_size > 0:
                    resin_orders.append(Order(symbol, price, order_size))
                    position += order_size

        # Calculate undercut prices (to improve on best current prices)
        undercut_buy = best_buy_price + 1  # One tick better than current best buy
        undercut_sell = best_sell_price - 1  # One tick better than current best sell
    
        # Calculate our bid and ask prices
        bid_price = min(undercut_buy, fair_value - 2)  # Don't bid above fair value - 1
        ask_price = max(undercut_sell, fair_value + 2)  # Don't ask below fair value + 1
    
        # STEP 2: Add extra buying pressure if we're short
        if position < self.position_limit and current_position < -10:
            extra_buy_size = min(45, self.position_limit - position)
            extra_buy_price = min(undercut_buy + 1, fair_value - 1)
            resin_orders.append(Order(symbol, extra_buy_price, extra_buy_size))
            position += extra_buy_size
    
        # STEP 3: Add cautious buying when we're very long
        if position < self.position_limit and current_position > 35:
            cautious_buy_size = min(40, self.position_limit - position)
            cautious_buy_price = min(undercut_buy - 1, fair_value - 1)  # Slightly worse price
            resin_orders.append(Order(symbol, cautious_buy_price, cautious_buy_size))
            position += cautious_buy_size
    
        # STEP 4: Add regular bid to maintain market making presence
        if position < self.position_limit:
            regular_buy_size = min(40, self.position_limit - position)
            resin_orders.append(Order(symbol, bid_price, regular_buy_size))
            position += regular_buy_size

        # Reset position tracking for sell orders
        position = current_position
    
        # STEP 5: Sell into any overpriced buy orders
        for price, volume in buy_orders.items():
            if (price > fair_value or (current_position > 0 and price == fair_value)) and position > -self.position_limit:
                order_size = max(-volume, -self.position_limit - position)
                if order_size < 0:  # Must be negative for sell orders
                    resin_orders.append(Order(symbol, price, order_size))
                    position += order_size
        # STEP 6: Add extra selling pressure if we're long
        if position > -self.position_limit and current_position > 35:
            extra_sell_size = max(-40, -self.position_limit - position)
            extra_sell_price = max(undercut_sell - 1, fair_value + 1)
            resin_orders.append(Order(symbol, extra_sell_price, extra_sell_size))
            position += extra_sell_size
    
        # STEP 7: Add cautious selling when we're very short
        if position > -self.position_limit and current_position < -35:
            cautious_sell_size = max(-40, -self.position_limit - position)
            cautious_sell_price = max(undercut_sell + 1, fair_value + 1)  # Slightly better price
            resin_orders.append(Order(symbol, cautious_sell_price, cautious_sell_size))
            position += cautious_sell_size
    
        # STEP 8: Add regular ask to maintain market making presence
        if position > -self.position_limit:
            regular_sell_size = max(-40, -self.position_limit - position)
            resin_orders.append(Order(symbol, ask_price, regular_sell_size))
            position += regular_sell_size

        
    
        # Add orders to result
        result[symbol] = resin_orders

    def process_kelp_regression(self, state: TradingState, result: dict) -> None:
        """
        Process KELP using regression model similar to BANANAS strategy
        """
        symbol = "KELP"
        position_limit = 50

        # Initialize KELP cache if not exists
        if not hasattr(self, 'kelp_cache'):
            self.kelp_cache = []

        # sort order books for calculating mid price
        order_depth = state.order_depths[symbol]
        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        # calculate mid price
        best_sell_price = min(sell_orders.keys()) if sell_orders else None
        best_buy_price = max(buy_orders.keys()) if buy_orders else None

        if best_sell_price and best_buy_price:
            mid_price = (best_sell_price + best_buy_price) / 2
        elif best_sell_price:
            mid_price = best_sell_price
            if len(self.kelp_cache) == self.kelp_dim:
                mid_price = (self.calc_next_price_kelp() + mid_price) / 2
        elif best_buy_price:
            mid_price = best_buy_price
            if len(self.kelp_cache) == self.kelp_dim:
                mid_price = (self.calc_next_price_kelp() + mid_price) / 2
        else:
            
            mid_price = self.last_mid_prices.get(symbol, 2000)  # 2k default

        # update kelp cache to remove stale vals
        if len(self.kelp_cache) == self.kelp_dim:
            self.kelp_cache.pop(0)
        self.kelp_cache.append(mid_price)

        # predict next price
        if len(self.kelp_cache) == self.kelp_dim:
            next_price = self.calc_next_price_kelp()
        else:
            next_price = mid_price

        # Define acceptable price bounds
        acc_bid = int(next_price - 1)
        acc_ask = int(next_price + 1)

        # compute orders using regression method
        kelp_orders = self.compute_orders_regression(
            symbol, state, acc_bid, acc_ask, position_limit
        )
        # store current mid price for future reference
        self.last_mid_prices[symbol] = mid_price

        # add orders to result
        result[symbol] = kelp_orders

    def calc_next_price_kelp(self):
        """
        Predict the next KELP price using a regression model - got coefficients from data analysis on cleaned data
        """
        
        coef = [0.198536, 0.204951, 0.260282, 0.335230]
        intercept = 2.024153
        
        # prediction using recent cache values
        nxt_price = intercept
        for i, val in enumerate(self.kelp_cache):
            nxt_price += val * coef[i]
        
        return int(round(nxt_price))

    def compute_orders_regression(
    self, 
    symbol: str, 
    state: TradingState, 
    acc_bid: int, 
    acc_ask: int, 
    LIMIT: int
):
        """
        Compute orders using a regression-based strategy
        
        :param symbol: Product symbol to trade
        :param state: Current trading state
        :param acc_bid: Acceptable bid price
        :param acc_ask: Acceptable ask price
        :param LIMIT: Position limit
        :return: List of orders
        """
        orders: list[Order] = []
        order_depth = state.order_depths[symbol]
        
        # Sort order books
        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        
        # Helper function to extract values
        def values_extract(order_dict, buy=0):
            if len(order_dict) == 0:
                return 0, -1 if buy else float('inf')
            best_price = max(order_dict.keys()) if buy else min(order_dict.keys())
            return abs(order_dict[best_price]), best_price

        # Extract best prices and volumes
        sell_vol, best_sell_pr = values_extract(osell)
        buy_vol, best_buy_pr = values_extract(obuy, 1)
        
        # Get current position for this symbol
        cpos = state.position.get(symbol, 0)

        # Buy underpriced sell orders
        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((cpos < 0) and (ask == acc_bid+1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                if order_for > 0:
                    orders.append(Order(symbol, ask, order_for))
                    cpos += order_for

        # Add limit buy order
        undercut_buy = best_buy_pr + 1 if best_buy_pr != -1 else acc_bid
        bid_pr = min(undercut_buy, acc_bid)
        
        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(symbol, bid_pr, num))
            cpos += num

        # Reset position tracking for sell orders
        cpos = state.position.get(symbol, 0)

        # Sell overpriced buy orders
        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((cpos > 0) and (bid+1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT-cpos)
                if order_for < 0:
                    orders.append(Order(symbol, bid, order_for))
                    cpos += order_for

        # Add limit sell order
        undercut_sell = best_sell_pr - 1 if best_sell_pr != float('inf') else acc_ask
        sell_pr = max(undercut_sell, acc_ask)
        
        if cpos > -LIMIT:
            num = -LIMIT-cpos
            orders.append(Order(symbol, sell_pr, num))
            cpos += num

        return orders
    


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
        if "RAINFOREST_RESIN" in state.order_depths:
            self.process_rainforest_resin_simple(state, result)
        
        # Process KELP with our linear regression model
        if "KELP" in state.order_depths:
            # Use our new regression model instead of the simple approach
            self.process_kelp_regression(state, result)
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData