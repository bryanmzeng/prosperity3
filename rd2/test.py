from rd2.datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
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


# Update the PARAMS dictionary to include all necessary parameters
PARAMS = {
    Product.SPREAD: {
        "default_spread_mean": 46.7624, #46.7624 at 7.8k, 48.7624 original
        "default_spread_std": 83.5354,
        "spread_std_window": 55, #55 best
        "base_zscore_threshold": 3, # Base threshold value
        "zscore_threshold": 3, # Keep for backward compatibility
        "max_zscore_threshold": 5, # Maximum threshold during volatile periods
        "volatility_scaling": True, # Enable dynamic thresholds
        "base_position": 58, # Add this - same as target_position
        "target_position": 58, # Keep for backward compatibility
        "min_position": 20,  # Add minimum position
        "max_position": 100, # Add maximum position
        "warmup_period": 100, # Initial period to observe before trading
        "max_drawdown_pct": 0.05, # Maximum acceptable drawdown (5%)
        "trend_detection": True, # Enable trend detection
        "trend_window": 20, # Window for trend detection
        "trend_threshold": 0.6, # Threshold for trend detection
    },
    Product.SPREAD2: {
        "default_spread_mean": 30.2359,
        "default_spread_std": 54.0495,
        "spread_std_window": 55, #55 best
        "base_zscore_threshold": 3, # Base threshold value
        "zscore_threshold": 3, # Keep for backward compatibility
        "max_zscore_threshold": 5, # Maximum threshold during volatile periods
        "volatility_scaling": True, # Enable dynamic thresholds
        "base_position": -55, # Add this - same as target_position
        "target_position": -55, # Keep for backward compatibility
        "min_position": -20, # Add minimum position (negative for short)
        "max_position": -100, # Add maximum position (negative for short)
        "warmup_period": 100, # Initial period to observe before trading
        "max_drawdown_pct": 0.05, # Maximum acceptable drawdown (5%)
        "trend_detection": True, # Enable trend detection
        "trend_window": 20, # Window for trend detection
        "trend_threshold": 0.6, # Threshold for trend detection
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


    def detect_trending_market(self, spread_history, product):
        """Detect if the market is in a strong trend (which could break mean reversion)"""
        params = self.params[product]
        
        # Need enough data for trend detection
        if len(spread_history) < params["trend_window"]:
            return False
        
        # Get the recent window of spread values
        recent_spreads = spread_history[-params["trend_window"]:]
        
        # Calculate directional movement
        ups = sum(1 for i in range(1, len(recent_spreads)) if recent_spreads[i] > recent_spreads[i-1])
        downs = sum(1 for i in range(1, len(recent_spreads)) if recent_spreads[i] < recent_spreads[i-1])
        
        # Calculate directional strength
        total_moves = ups + downs
        if total_moves == 0:
            return False
            
        # If movement is predominantly in one direction, it's trending
        directional_strength = max(ups, downs) / total_moves
        return directional_strength > params["trend_threshold"]
    def calculate_spread_drawdown(self, initial_spread, current_spread):
        """Calculate drawdown percentage from initial spread"""
        if initial_spread == 0:
            return 0
        return abs((current_spread - initial_spread) / initial_spread)
    def should_enter_trade(self, product, spread_data, timestamp):
        """Determine if we should enter a trade based on various risk factors"""
        params = self.params[product]
        
        # 1. Skip trading during initial warmup period
        if timestamp < params.get("warmup_period", 0):
            return False
            
        # 2. Check if market is in a strong trend (avoid mean reversion during trends)
        if params.get("trend_detection", False) and self.detect_trending_market(spread_data["spread_history"], product):
            return False
        
        # 3. If we've recently experienced a large drawdown, be more cautious
        if "initial_spread" in spread_data and "max_drawdown_pct" in params:
            current_spread = spread_data["spread_history"][-1]
            drawdown = self.calculate_spread_drawdown(spread_data["initial_spread"], current_spread)
            if drawdown > params["max_drawdown_pct"]:
                return False
        
        return True
    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )

    def calculate_dynamic_position(self, product, zscore, spread_std):
        """Calculate dynamic position size based on z-score magnitude and spread volatility"""
        params = self.params[product]
        base_position = params["base_position"]
        
        # Scale based on z-score magnitude (stronger signal = larger position)
        zscore_magnitude = abs(zscore) / params["zscore_threshold"]
        position_scale = min(max(zscore_magnitude, 0.5), 2.0)  # Range: 0.5x to 2.0x of base
        
        # Adjust for volatility (higher volatility = smaller position)
        volatility_ratio = params["default_spread_std"] / max(spread_std, 1e-6)
        volatility_scale = min(max(volatility_ratio, 0.5), 1.5)  # Range: 0.5x to 1.5x
        
        # Calculate target position (preserving direction)
        sign = 1 if base_position > 0 else -1
        scaled_position = int(abs(base_position) * position_scale * volatility_scale) * sign
        
        # Ensure position is within allowed range
        if sign > 0:
            return max(min(scaled_position, params["max_position"]), params["min_position"])
        else:
            return min(max(scaled_position, params["max_position"]), params["min_position"])

    def calculate_dynamic_zscore_threshold(self, product, spread_std):
        """Calculate dynamic z-score threshold based on current volatility"""
        params = self.params[product]
        
        # If volatility scaling is disabled, return the base threshold
        if not params.get("volatility_scaling", False):
            return params["base_zscore_threshold"]
        
        # Calculate volatility ratio (current vs. historical)
        volatility_ratio = spread_std / params["default_spread_std"]
        
        # Scale threshold based on volatility ratio
        # Higher volatility = higher threshold = fewer trades
        if volatility_ratio > 1.5:
            # Very high volatility - use maximum threshold
            return params["max_zscore_threshold"]
        elif volatility_ratio > 1.0:
            # Moderate to high volatility - scale between base and max
            scaling_factor = (volatility_ratio - 1.0) / 0.5  # 0 to 1 scaling
            return params["base_zscore_threshold"] + scaling_factor * (params["max_zscore_threshold"] - params["base_zscore_threshold"])
        else:
            # Normal or low volatility - use base threshold
            return params["base_zscore_threshold"]

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
        timestamp: int = 0,
    ):
        if Product.PICNIC_BASKET1 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        SYNTHETIC1_order_depth = self.get_SYNTHETIC1_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        SYNTHETIC1_swmid = self.get_swmid(SYNTHETIC1_order_depth)
        spread = basket_swmid - SYNTHETIC1_swmid
        spread_data["spread_history"].append(spread)
        
        # Record initial spread value if not set
        if "initial_spread" not in spread_data:
            spread_data["initial_spread"] = spread

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])
        
        # Calculate dynamic z-score threshold
        current_threshold = self.calculate_dynamic_zscore_threshold(Product.SPREAD, spread_std)
        
        zscore = (
            spread - self.params[Product.SPREAD]["default_spread_mean"]
        ) / spread_std

        # Check if we should be trading based on risk factors
        if not self.should_enter_trade(Product.SPREAD, spread_data, timestamp):
            # Just update tracking values but don't trade
            spread_data["prev_zscore"] = zscore
            return None

        # Updated to use current_threshold instead of zscore_threshold
        if zscore >= current_threshold:
            if basket_position != -self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(
                    -self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -current_threshold:
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
        timestamp: int = 0,
    ):
        if Product.PICNIC_BASKET2 not in order_depths:
            return None

        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        SYNTHETIC2_order_depth = self.get_SYNTHETIC2_basket_order_depth(order_depths)

        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(SYNTHETIC2_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)
        
        # Record initial spread value if not set
        if "initial_spread" not in spread_data:
            spread_data["initial_spread"] = spread

        if len(spread_data["spread_history"]) < self.params[Product.SPREAD2]["spread_std_window"]:
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD2]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])
        
        # Calculate dynamic z-score threshold
        current_threshold = self.calculate_dynamic_zscore_threshold(Product.SPREAD2, spread_std)
        
        zscore = (spread - self.params[Product.SPREAD2]["default_spread_mean"]) / spread_std

        # Check if we should be trading based on risk factors
        if not self.should_enter_trade(Product.SPREAD2, spread_data, timestamp):
            # Just update tracking values but don't trade
            spread_data["prev_zscore"] = zscore
            return None

        # Updated to use current_threshold instead of zscore_threshold
        if zscore >= current_threshold:
            if basket_position != -self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(
                    -self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -current_threshold:
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

        # Initialize spread data if needed
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

        # Get positions
        basket_position = (
            state.position[Product.PICNIC_BASKET1]
            if Product.PICNIC_BASKET1 in state.position
            else 0
        )
        
        basket2_position = (
            state.position.get(Product.PICNIC_BASKET2, 0)
        )

        # Pass timestamp to the spread_orders functions
        timestamp = state.timestamp if hasattr(state, 'timestamp') else 0
        
        # Get trading signals with timestamp
        spread1_orders = self.spread_orders(
            state.order_depths,
            Product.PICNIC_BASKET1,
            basket_position,
            traderObject[Product.SPREAD],
            timestamp
        )
        
        spread2_orders = self.spread2_orders(
            state.order_depths,
            Product.PICNIC_BASKET2,
            basket2_position,
            traderObject[Product.SPREAD2],
            timestamp
        )

        # Rest of your arbitrage prioritization logic...
        arb_candidates = []

        if spread1_orders:
            arb_candidates.append(('ETF1', spread1_orders, abs(traderObject[Product.SPREAD]["prev_zscore"])))

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