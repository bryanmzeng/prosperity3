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
        "default_spread_mean": 46.7624,
        "default_spread_std": 83.5354,
        "spread_std_window": 58,
        "zscore_threshold": 3,
        "target_position": 55,
    },
    Product.SPREAD2: {
        "default_spread_mean": 30.2359,
        "default_spread_std": 54.0495,
        "spread_std_window": 55,
        "zscore_threshold": 3,
        "target_position": -85,
    },
    # Market making parameters for component products
    Product.CROISSANTS: {
        "take_width": 2,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 5,
        "soft_position_limit": 150,
        "hedge_threshold": 14,
        "hedge_ratio": 0.65
    },
    Product.JAMS: {
        "take_width": 2,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 5,
        "soft_position_limit": 200,
        "hedge_threshold": 7,
        "hedge_ratio": 0.65
    },
    Product.DJEMBES: {
        "take_width": 3,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 6,
        "soft_position_limit": 40,
        "hedge_threshold": 4,
        "hedge_ratio": 0.75
    },
    "JAMS_DJEMBES_PAIR": {
        "default_ratio_mean": 0.49,  # Expected mean of Jams/Djembes price ratio
        "default_ratio_std": 0.015,  # Standard deviation of the ratio
        "ratio_std_window": 50,      # Window for calculating ratio std
        "zscore_threshold": 3,     # Z-score threshold for trading
        "position_limit":60,        # Max position in either leg
        "target_position": 58,       # Target position to take when signal triggers
        "min_edge": 2,               # Minimum edge for taking trades
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
        # Configuration parameters
        self.position_limit = 50
        self.target_inventory = 20 # Target inventory level (default: neutral)
        
        
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
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def get_mid(self, order_depth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

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
        
        # Check if either mid is None (no valid orders on one side)
        if basket_swmid is None or SYNTHETIC1_swmid is None:
            return None
            
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
        
        # Check if either mid is None (no valid orders on one side)
        if basket_swmid is None or synthetic_swmid is None:
            return None
            
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
    
    # Market making functions from the second code snippet
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if best_ask <= fair_value - take_width:
                quantity = min(
                    best_ask_amount, position_limit - position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if best_bid >= fair_value + take_width:
                quantity = min(
                    best_bid_amount, position_limit + position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    # Calculate fair value for each of the components based on the arbitrage spreads
    def calculate_fair_values(self, state: TradingState, traderObject):
        fair_values = {}
        
        # Initialize storage for fair values in trader object if not present
        if "fair_values" not in traderObject:
            traderObject["fair_values"] = {}
        
        # First, try to calculate fair values from ETF arbs
        if Product.PICNIC_BASKET1 in state.order_depths and all(p in state.order_depths for p in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]):
            basket1_mid = self.get_mid(state.order_depths[Product.PICNIC_BASKET1])
            synthetic1_order_depth = self.get_SYNTHETIC1_basket_order_depth(state.order_depths)
            synthetic1_mid = self.get_mid(synthetic1_order_depth)
            
            if basket1_mid is not None and synthetic1_mid is not None:
                # Average of ETF and synthetic gives us a good fair value estimate
                basket1_fair = (basket1_mid + synthetic1_mid) / 2
                traderObject["fair_values"]["BASKET1"] = basket1_fair
                
                # Derive component fair values based on the basket weights
                total_weight = sum(BASKET_WEIGHTS.values())
                # We need relative weights to decompose the basket fair value
                weight_ratio = {
                    k: v / total_weight for k, v in BASKET_WEIGHTS.items()
                }
                
                # If we have historical fair values, use them as base
                if all(p in traderObject["fair_values"] for p in BASKET_WEIGHTS.keys()):
                    # Use historical relative values and adjust for new basket fair value
                    current_total = sum(traderObject["fair_values"][p] * BASKET_WEIGHTS[p] for p in BASKET_WEIGHTS)
                    adjustment_factor = basket1_fair / (current_total / total_weight)
                    
                    for product in BASKET_WEIGHTS:
                        fair_values[product] = traderObject["fair_values"][product] * adjustment_factor
                else:
                    # Initial estimate based on order book
                    component_mids = {}
                    for product in BASKET_WEIGHTS:
                        if product in state.order_depths:
                            mid = self.get_mid(state.order_depths[product])
                            if mid is not None:
                                component_mids[product] = mid
                    
                    if len(component_mids) == len(BASKET_WEIGHTS):
                        # Scale components to match basket fair value
                        current_basket_value = sum(component_mids[p] * BASKET_WEIGHTS[p] for p in BASKET_WEIGHTS)
                        adjustment_factor = basket1_fair / (current_basket_value / total_weight)
                        
                        for product in BASKET_WEIGHTS:
                            fair_values[product] = component_mids[product] * adjustment_factor
        
        # Also check ETF2 for additional data points
        if Product.PICNIC_BASKET2 in state.order_depths and all(p in state.order_depths for p in BASKET2_WEIGHTS.keys()):
            basket2_mid = self.get_mid(state.order_depths[Product.PICNIC_BASKET2])
            synthetic2_order_depth = self.get_SYNTHETIC2_basket_order_depth(state.order_depths)
            synthetic2_mid = self.get_mid(synthetic2_order_depth)
            
            if basket2_mid is not None and synthetic2_mid is not None:
                basket2_fair = (basket2_mid + synthetic2_mid) / 2
                traderObject["fair_values"]["BASKET2"] = basket2_fair
                
                # If we don't have fair values from ETF1, calculate from ETF2
                if not fair_values:
                    total_weight = sum(BASKET2_WEIGHTS.values())
                    weight_ratio = {
                        k: v / total_weight for k, v in BASKET2_WEIGHTS.items()
                    }
                    
                    # Similar logic as with ETF1
                    if all(p in traderObject["fair_values"] for p in BASKET2_WEIGHTS.keys()):
                        current_total = sum(traderObject["fair_values"][p] * BASKET2_WEIGHTS[p] for p in BASKET2_WEIGHTS)
                        adjustment_factor = basket2_fair / (current_total / total_weight)
                        
                        for product in BASKET2_WEIGHTS:
                            fair_values[product] = traderObject["fair_values"][product] * adjustment_factor
                    else:
                        component_mids = {}
                        for product in BASKET2_WEIGHTS:
                            if product in state.order_depths:
                                mid = self.get_mid(state.order_depths[product])
                                if mid is not None:
                                    component_mids[product] = mid
                        
                        if len(component_mids) == len(BASKET2_WEIGHTS):
                            current_basket_value = sum(component_mids[p] * BASKET2_WEIGHTS[p] for p in BASKET2_WEIGHTS)
                            adjustment_factor = basket2_fair / (current_basket_value / total_weight)
                            
                            for product in BASKET2_WEIGHTS:
                                fair_values[product] = component_mids[product] * adjustment_factor
        
        # If we couldn't calculate from ETFs, fall back to mid prices
        for product in set(list(BASKET_WEIGHTS.keys()) + list(BASKET2_WEIGHTS.keys())):
            if product not in fair_values and product in state.order_depths:
                mid = self.get_mid(state.order_depths[product])
                if mid is not None:
                    fair_values[product] = mid
        
        # Update stored fair values
        for product, value in fair_values.items():
            traderObject["fair_values"][product] = value
        
        return fair_values, traderObject
    def calculate_jams_djembes_ratio(self, state: TradingState, traderObject):
        """
        Calculate the current price ratio between Jams and Djembes and update history.
        Returns the current ratio and z-score.
        """
        # Initialize ratio data if not present
        if "JAMS_DJEMBES_RATIO" not in traderObject:
            traderObject["JAMS_DJEMBES_RATIO"] = {
                "ratio_history": [],
                "prev_zscore": 0,
            }
        
        # Get current mid prices
        jams_depth = state.order_depths.get(Product.JAMS)
        djembes_depth = state.order_depths.get(Product.DJEMBES)
        
        if jams_depth is None or djembes_depth is None:
            return None, None, traderObject
            
        jams_mid = self.get_mid(jams_depth)
        djembes_mid = self.get_mid(djembes_depth)
        
        if jams_mid is None or djembes_mid is None:
            return None, None, traderObject
        
        # Calculate ratio (Jams / Djembes)
        current_ratio = jams_mid / djembes_mid
        
        # Update ratio history
        traderObject["JAMS_DJEMBES_RATIO"]["ratio_history"].append(current_ratio)
        
        # Maintain window size
        window_size = self.params["JAMS_DJEMBES_PAIR"]["ratio_std_window"]
        if len(traderObject["JAMS_DJEMBES_RATIO"]["ratio_history"]) > window_size:
            traderObject["JAMS_DJEMBES_RATIO"]["ratio_history"].pop(0)
        
        # If we don't have enough history, use default values
        if len(traderObject["JAMS_DJEMBES_RATIO"]["ratio_history"]) < window_size * 0.5:
            ratio_mean = self.params["JAMS_DJEMBES_PAIR"]["default_ratio_mean"]
            ratio_std = self.params["JAMS_DJEMBES_PAIR"]["default_ratio_std"]
        else:
            # Calculate mean and std from history
            ratio_mean = np.mean(traderObject["JAMS_DJEMBES_RATIO"]["ratio_history"])
            ratio_std = np.std(traderObject["JAMS_DJEMBES_RATIO"]["ratio_history"])
            
            # Blend with default values for stability
            default_mean = self.params["JAMS_DJEMBES_PAIR"]["default_ratio_mean"]
            default_std = self.params["JAMS_DJEMBES_PAIR"]["default_ratio_std"]
            
            # Gradually increase weight of calculated values as we get more data
            data_weight = min(1.0, len(traderObject["JAMS_DJEMBES_RATIO"]["ratio_history"]) / window_size)
            ratio_mean = (ratio_mean * data_weight) + (default_mean * (1 - data_weight))
            ratio_std = (ratio_std * data_weight) + (default_std * (1 - data_weight))
        
        # Calculate z-score
        if ratio_std > 0:
            zscore = (current_ratio - ratio_mean) / ratio_std
        else:
            zscore = 0
            
        # Store current zscore
        traderObject["JAMS_DJEMBES_RATIO"]["prev_zscore"] = zscore
        
        return current_ratio, zscore, traderObject
        
    def execute_jams_djembes_mean_reversion(self, state: TradingState, traderObject, ratio, zscore):
        """
        Execute mean reversion trades between Jams and Djembes based on ratio z-score.
        """
        if ratio is None or zscore is None:
            return {}, traderObject
            
        result = {}
        
        # Get current positions
        jams_position = state.position.get(Product.JAMS, 0)
        djembes_position = state.position.get(Product.DJEMBES, 0)
        
        # Get parameters
        zscore_threshold = self.params["JAMS_DJEMBES_PAIR"]["zscore_threshold"]
        position_limit = self.params["JAMS_DJEMBES_PAIR"]["position_limit"]
        target_position = self.params["JAMS_DJEMBES_PAIR"]["target_position"]
        min_edge = self.params["JAMS_DJEMBES_PAIR"]["min_edge"]
        
        # Check if we can take new positions
        can_short_jams = jams_position > -self.LIMIT[Product.JAMS] + target_position
        can_long_jams = jams_position < self.LIMIT[Product.JAMS] - target_position
        can_short_djembes = djembes_position > -self.LIMIT[Product.DJEMBES] + target_position
        can_long_djembes = djembes_position < self.LIMIT[Product.DJEMBES] - target_position
        
        # Get best prices
        jams_depth = state.order_depths.get(Product.JAMS)
        djembes_depth = state.order_depths.get(Product.DJEMBES)
        
        if jams_depth is None or djembes_depth is None:
            return {}, traderObject
            
        if not jams_depth.buy_orders or not jams_depth.sell_orders or not djembes_depth.buy_orders or not djembes_depth.sell_orders:
            return {}, traderObject
            
        jams_best_bid = max(jams_depth.buy_orders.keys())
        jams_best_ask = min(jams_depth.sell_orders.keys())
        djembes_best_bid = max(djembes_depth.buy_orders.keys())
        djembes_best_ask = min(djembes_depth.sell_orders.keys())
        
        # Initialize orders
        jams_orders = []
        djembes_orders = []
        
        # If ratio is too high (Jams overvalued / Djembes undervalued)
        if zscore > zscore_threshold and can_short_jams and can_long_djembes:
            # Calculate available volumes
            jams_bid_volume = abs(jams_depth.buy_orders[jams_best_bid])
            djembes_ask_volume = abs(djembes_depth.sell_orders[djembes_best_ask])
            
            # Check if we have enough edge
            current_edge = jams_best_bid / djembes_best_ask - ratio
            if current_edge < min_edge:
                return {}, traderObject
                
            # Calculate trade size
            jams_capacity = min(jams_bid_volume, self.LIMIT[Product.JAMS] + jams_position, position_limit)
            djembes_capacity = min(djembes_ask_volume, self.LIMIT[Product.DJEMBES] - djembes_position, position_limit)
            
            # Ensure balanced positions
            if jams_capacity > 0 and djembes_capacity > 0:
                trade_size = min(target_position, jams_capacity, djembes_capacity)
                
                # Create orders
                jams_orders.append(Order(Product.JAMS, jams_best_bid, -trade_size))
                djembes_orders.append(Order(Product.DJEMBES, djembes_best_ask, trade_size))
        
        # If ratio is too low (Jams undervalued / Djembes overvalued)
        elif zscore < -zscore_threshold and can_long_jams and can_short_djembes:
            # Calculate available volumes
            jams_ask_volume = abs(jams_depth.sell_orders[jams_best_ask])
            djembes_bid_volume = abs(djembes_depth.buy_orders[djembes_best_bid])
            
            # Check if we have enough edge
            current_edge = ratio - jams_best_ask / djembes_best_bid
            if current_edge < min_edge:
                return {}, traderObject
                
            # Calculate trade size
            jams_capacity = min(jams_ask_volume, self.LIMIT[Product.JAMS] - jams_position, position_limit)
            djembes_capacity = min(djembes_bid_volume, self.LIMIT[Product.DJEMBES] + djembes_position, position_limit)
            
            # Ensure balanced positions
            if jams_capacity > 0 and djembes_capacity > 0:
                trade_size = min(target_position, jams_capacity, djembes_capacity)
                
                # Create orders
                jams_orders.append(Order(Product.JAMS, jams_best_ask, trade_size))
                djembes_orders.append(Order(Product.DJEMBES, djembes_best_bid, -trade_size))
        
        # Check for mean reversion - close positions
        elif abs(zscore) < 0.5:
            # If we have positions, start unwinding them
            if jams_position > 10 and djembes_position < -10:
                # We are long Jams, short Djembes
                jams_sell_size = min(abs(jams_position), jams_depth.buy_orders[jams_best_bid])
                djembes_buy_size = min(abs(djembes_position), abs(djembes_depth.sell_orders[djembes_best_ask]))
                
                unwind_size = min(jams_sell_size, djembes_buy_size, 10)  # Limit unwind to 10 units per step
                
                if unwind_size > 0:
                    jams_orders.append(Order(Product.JAMS, jams_best_bid, -unwind_size))
                    djembes_orders.append(Order(Product.DJEMBES, djembes_best_ask, unwind_size))
                    
            elif jams_position < -10 and djembes_position > 10:
                # We are short Jams, long Djembes
                jams_buy_size = min(abs(jams_position), abs(jams_depth.sell_orders[jams_best_ask]))
                djembes_sell_size = min(abs(djembes_position), djembes_depth.buy_orders[djembes_best_bid])
                
                unwind_size = min(jams_buy_size, djembes_sell_size, 10)  # Limit unwind to 10 units per step
                
                if unwind_size > 0:
                    jams_orders.append(Order(Product.JAMS, jams_best_ask, unwind_size))
                    djembes_orders.append(Order(Product.DJEMBES, djembes_best_bid, -unwind_size))
        
        # Add orders to result if not empty
        if jams_orders:
            result[Product.JAMS] = jams_orders
        if djembes_orders:
            result[Product.DJEMBES] = djembes_orders
            
        return result, traderObject
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

        # Initialize storage for spread history
        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        if Product.SPREAD2 not in traderObject:
            traderObject[Product.SPREAD2] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        # Get current positions
        basket_position = state.position.get(Product.PICNIC_BASKET1, 0)
        basket2_position = state.position.get(Product.PICNIC_BASKET2, 0)
        
        # Calculate fair values for all products
        fair_values, traderObject = self.calculate_fair_values(state, traderObject)
        jams_djembes_ratio, ratio_zscore, traderObject = self.calculate_jams_djembes_ratio(state, traderObject)
        
        

        # Check for arbitrage opportunities
        arb_candidates = []

        # ETF1 arb
        if Product.PICNIC_BASKET1 in state.order_depths:
            spread1_orders = self.spread_orders(
                state.order_depths,
                Product.PICNIC_BASKET1,
                basket_position,
                traderObject[Product.SPREAD],
            )
            if spread1_orders:
                arb_candidates.append(('ETF1', spread1_orders, abs(traderObject[Product.SPREAD].get("prev_zscore", 0))))

        # ETF2 arb
        if Product.PICNIC_BASKET2 in state.order_depths:
            spread2_orders = self.spread2_orders(
                state.order_depths,
                Product.PICNIC_BASKET2,
                basket2_position,
                traderObject[Product.SPREAD2],
            )
            if spread2_orders:
                arb_candidates.append(('ETF2', spread2_orders, abs(traderObject[Product.SPREAD2].get("prev_zscore", 0))))
        if jams_djembes_ratio is not None and ratio_zscore is not None:
            # Only add if the signal is strong enough
            # if abs(ratio_zscore) > self.params["JAMS_DJEMBES_PAIR"]["zscore_threshold"]:
            #     jams_djembes_orders, traderObject = self.execute_jams_djembes_mean_reversion(state, traderObject, jams_djembes_ratio, ratio_zscore)
            #     if jams_djembes_orders:
            #         arb_candidates.append(('JAMS_DJEMBES_PAIR', jams_djembes_orders, abs(ratio_zscore)))
            jams_djembes_orders, traderObject = self.execute_jams_djembes_mean_reversion(state, traderObject, jams_djembes_ratio, ratio_zscore)
            if jams_djembes_orders:
                arb_candidates.append(('JAMS_DJEMBES_PAIR', jams_djembes_orders, abs(ratio_zscore)))
        

        # Sort by z-score magnitude (or expected edge)
        arb_candidates.sort(key=lambda x: x[2], reverse=True)

        # Pick top priority arbitrage
        if arb_candidates:
            chosen_arb = arb_candidates[0]
            arb_id, arb_orders, _ = chosen_arb
            if arb_id == 'ETF1':
                for product in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES, Product.PICNIC_BASKET1]:
                    if product in arb_orders:
                        result[product] = arb_orders[product]
            elif arb_id == 'ETF2':
                for product in [Product.CROISSANTS, Product.JAMS, Product.PICNIC_BASKET2]:
                    if product in arb_orders:
                        result[product] = arb_orders[product]
            elif arb_id == 'JAMS_DJEMBES_PAIR':
                for product in [Product.DJEMBES, Product.JAMS]:
                    if product in arb_orders:
                        result[product] = arb_orders[product]
        
        # Hedging for arbitrage positions
        # Only apply hedging if we're in an arbitrage position
        if arb_candidates:
            chosen_arb = arb_candidates[0]
            arb_id, _, zscore = chosen_arb
            
            # Track net component exposure from arb positions
            component_exposure = {
                Product.CROISSANTS: 0,
                Product.JAMS: 0,
                Product.DJEMBES: 0
            }
            
            # Calculate current exposures from positions
            for product in component_exposure:
                component_exposure[product] = state.position.get(product, 0)
            
            # Calculate ETF exposures and implied component exposure
            basket1_exposure = state.position.get(Product.PICNIC_BASKET1, 0)
            basket2_exposure = state.position.get(Product.PICNIC_BASKET2, 0)
            
            # Add the implied exposure from ETF1
            for product, weight in BASKET_WEIGHTS.items():
                component_exposure[product] += basket1_exposure * weight
            
            # Add the implied exposure from ETF2 (only affects CROISSANTS and JAMS)
            for product, weight in BASKET2_WEIGHTS.items():
                component_exposure[product] += basket2_exposure * weight
            
            # Now hedge the component exposure
            for product, exposure in component_exposure.items():
                # Skip if we already have orders for this product from arb
                # if product in result:
                #     continue
                
                # Only hedge if exposure exceeds the threshold
                hedge_threshold = self.params[product]["hedge_threshold"]
                hedge_ratio = self.params[product]["hedge_ratio"]
                
                if abs(exposure) > hedge_threshold:
                    position = state.position.get(product, 0)
                    
                    # Calculate the amount to hedge based on hedge ratio
                    target_hedge_amount = int(exposure * hedge_ratio)
                    
                    # Copy order depth to avoid modifying original
                    order_depth_copy = OrderDepth()
                    if product in state.order_depths:
                        order_depth_copy.buy_orders = dict(state.order_depths[product].buy_orders)
                        order_depth_copy.sell_orders = dict(state.order_depths[product].sell_orders)
                    else:
                        continue  # Skip if no order book available
                    
                    orders = []
                    
                    # If we have positive exposure, place sell orders to hedge
                    if exposure > 0:
                        if order_depth_copy.buy_orders:
                            best_bid = max(order_depth_copy.buy_orders.keys())
                            best_bid_volume = order_depth_copy.buy_orders[best_bid]
                            hedge_quantity = min(best_bid_volume, abs(target_hedge_amount))
                            
                            if hedge_quantity > 0:
                                orders.append(Order(product, best_bid, -hedge_quantity))
                    
                    # If we have negative exposure, place buy orders to hedge
                    elif exposure < 0:
                        if order_depth_copy.sell_orders:
                            best_ask = min(order_depth_copy.sell_orders.keys())
                            best_ask_volume = abs(order_depth_copy.sell_orders[best_ask])
                            hedge_quantity = min(best_ask_volume, abs(target_hedge_amount))
                            
                            if hedge_quantity > 0:
                                orders.append(Order(product, best_ask, hedge_quantity))
                    
                    # Additionally, place limit orders for any remaining hedge amount
                    remaining_hedge = abs(target_hedge_amount) - (hedge_quantity if 'hedge_quantity' in locals() else 0)
                    
                    if remaining_hedge > 0:
                        fair_value = fair_values.get(product)
                        
                        if fair_value is not None:
                            if exposure > 0:  # Sell orders to hedge positive exposure
                                # Place limit order at a competitive price
                                limit_price = round(fair_value + self.params[product]["default_edge"])
                                orders.append(Order(product, limit_price, -remaining_hedge))
                            else:  # Buy orders to hedge negative exposure
                                limit_price = round(fair_value - self.params[product]["default_edge"])
                                orders.append(Order(product, limit_price, remaining_hedge))
                    
                    if orders:
                        result[product] = orders
        if "RAINFOREST_RESIN" in state.order_depths:
            self.process_rainforest_resin_simple(state, result)
        
        # Process KELP with our linear regression model
        if "KELP" in state.order_depths:
            # Use our new regression model instead of the simple approach
            self.process_kelp_regression(state, result)

        # Encode trader data for the next round
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData