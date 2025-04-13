from rd2.datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math
import statistics


class Product:
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"


PARAMS = {
    Product.SPREAD: {
        "default_spread_mean": 48.7624,  # Average of means across days: (70.0449 + 43.9226 + 32.3198) / 3
        "default_spread_std": 83.5354,   # Average of std devs across days: (78.6671 + 82.7626 + 89.1765) / 3
        "spread_std_window": 50,   # Larger window for rolling standard deviation to reduce sensitivity
        "zscore_threshold_upper": 2.0,  # Threshold for selling the spread
        "zscore_threshold_lower": -2.0, # Threshold for buying the spread
        "exit_buffer": 0.5,        # Buffer for exiting positions
        "target_position": 58,     # Target position size when taking positions
    },
}

BASKET_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS

        self.params = params

        self.LIMIT = {
            Product.PICNIC_BASKET1: 60,
            Product.CROISSANTS: 250,  # As specified in limits 
            Product.JAMS: 350,        # As specified in limits
            Product.DJEMBES: 60,      # As in limits
        }
        
    

    def get_swmid(self, order_depth: OrderDepth) -> float:
        """Calculate the size-weighted mid price from the order book"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )
    
    def get_synthetic_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        """Calculate the synthetic basket order depth from component order depths"""
        # Constants
        CROISSANTS_PER_BASKET = BASKET_WEIGHTS[Product.CROISSANTS]
        JAMS_PER_BASKET = BASKET_WEIGHTS[Product.JAMS]
        DJEMBES_PER_BASKET = BASKET_WEIGHTS[Product.DJEMBES]

        # Initialize the synthetic basket order depth
        synthetic_order_depth = OrderDepth()

        # Calculate the best bid and ask for each component
        croissants_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        croissants_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        jams_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        jams_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        djembes_best_bid = (
            max(order_depths[Product.DJEMBES].buy_orders.keys())
            if order_depths[Product.DJEMBES].buy_orders
            else 0
        )
        djembes_best_ask = (
            min(order_depths[Product.DJEMBES].sell_orders.keys())
            if order_depths[Product.DJEMBES].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            croissants_best_bid * CROISSANTS_PER_BASKET
            + jams_best_bid * JAMS_PER_BASKET
            + djembes_best_bid * DJEMBES_PER_BASKET
        )
        implied_ask = (
            croissants_best_ask * CROISSANTS_PER_BASKET
            + jams_best_ask * JAMS_PER_BASKET
            + djembes_best_ask * DJEMBES_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            croissants_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid]
                // CROISSANTS_PER_BASKET
            )
            jams_bid_volume = (
                order_depths[Product.JAMS].buy_orders[jams_best_bid]
                // JAMS_PER_BASKET
            )
            djembes_bid_volume = (
                order_depths[Product.DJEMBES].buy_orders[djembes_best_bid]
                // DJEMBES_PER_BASKET
            )
            implied_bid_volume = min(
                croissants_bid_volume, jams_bid_volume, djembes_bid_volume
            )
            synthetic_order_depth.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            croissants_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask]
                // CROISSANTS_PER_BASKET
            )
            jams_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[jams_best_ask]
                // JAMS_PER_BASKET
            )
            djembes_ask_volume = (
                -order_depths[Product.DJEMBES].sell_orders[djembes_best_ask]
                // DJEMBES_PER_BASKET
            )
            implied_ask_volume = min(
                croissants_ask_volume, jams_ask_volume, djembes_ask_volume
            )
            synthetic_order_depth.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_depth
    
    def convert_synthetic_basket_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        """Convert synthetic basket orders to component orders"""
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                croissants_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                jams_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
                djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            croissants_order = Order(
                Product.CROISSANTS,
                croissants_price,
                quantity * BASKET_WEIGHTS[Product.CROISSANTS],
            )
            jams_order = Order(
                Product.JAMS,
                jams_price,
                quantity * BASKET_WEIGHTS[Product.JAMS],
            )
            djembes_order = Order(
                Product.DJEMBES, djembes_price, quantity * BASKET_WEIGHTS[Product.DJEMBES]
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(croissants_order)
            component_orders[Product.JAMS].append(jams_order)
            component_orders[Product.DJEMBES].append(djembes_order)

        return component_orders
        
    def execute_spread_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):
        """Execute spread orders by trading the basket and its components"""
        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)

        if target_position > basket_position:
            # Buy basket, sell components
            if not basket_order_depth.sell_orders or not synthetic_order_depth.buy_orders:
                return None
                
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            
            if execute_volume <= 0:
                return None

            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders

        else:
            # Sell basket, buy components
            if not basket_order_depth.buy_orders or not synthetic_order_depth.sell_orders:
                return None
                
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)
            
            if execute_volume <= 0:
                return None

            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders

    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        """Generate orders based on the spread between the basket and synthetic basket"""
        if Product.PICNIC_BASKET1 not in order_depths.keys():
            return None

        # Check if all components are available
        for component in BASKET_WEIGHTS.keys():
            if component not in order_depths.keys():
                return None

        # Get order depths
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        
        # Calculate spread using size-weighted mid prices
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        
        if basket_swmid is None or synthetic_swmid is None:
            return None
            
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        # Check if we have enough data and maintain window size
        if len(spread_data["spread_history"]) < self.params[Product.SPREAD]["spread_std_window"]:
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD]["spread_std_window"]:
            spread_data["spread_history"].pop(0)
            
        # Calculate standard deviation and z-score
        spread_std = np.std(spread_data["spread_history"])
        
        # Prevent division by zero
        if spread_std == 0:
            spread_std = 0.0001
            
        zscore = (
            spread - self.params[Product.SPREAD]["default_spread_mean"]
        ) / spread_std

        print(f"Spread: {spread:.2f}, Z-Score: {zscore:.2f}, Position: {basket_position}")
        
        # Trading signals based on z-score thresholds
        if zscore >= self.params[Product.SPREAD]["zscore_threshold_upper"]:
            if basket_position != -self.params[Product.SPREAD]["target_position"]:
                print(f"SELL SIGNAL: z-score {zscore:.2f} > {self.params[Product.SPREAD]['zscore_threshold_upper']}")
                return self.execute_spread_orders(
                    -self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= self.params[Product.SPREAD]["zscore_threshold_lower"]:
            if basket_position != self.params[Product.SPREAD]["target_position"]:
                print(f"BUY SIGNAL: z-score {zscore:.2f} < {self.params[Product.SPREAD]['zscore_threshold_lower']}")
                return self.execute_spread_orders(
                    self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )
                
        # Exit positions when z-score reverts
        if basket_position > 0 and zscore > self.params[Product.SPREAD]["zscore_threshold_lower"] + self.params[Product.SPREAD]["exit_buffer"]:
            print(f"EXIT LONG: z-score {zscore:.2f} > {self.params[Product.SPREAD]['zscore_threshold_lower'] + self.params[Product.SPREAD]['exit_buffer']}")
            return self.execute_spread_orders(
                0,  # Target position of zero (full exit)
                basket_position,
                order_depths,
            )
            
        if basket_position < 0 and zscore < self.params[Product.SPREAD]["zscore_threshold_upper"] - self.params[Product.SPREAD]["exit_buffer"]:
            print(f"EXIT SHORT: z-score {zscore:.2f} < {self.params[Product.SPREAD]['zscore_threshold_upper'] - self.params[Product.SPREAD]['exit_buffer']}")
            return self.execute_spread_orders(
                0,  # Target position of zero (full exit)
                basket_position,
                order_depths,
            )

        spread_data["prev_zscore"] = zscore
        return None
        


    def calculate_effective_position(self, positions: Dict[str, int]) -> int:
        """
        Calculate the effective position in the spread
        Positive means long basket (short components), negative means short basket (long components)
        """
        basket_position = positions.get(Product.PICNIC_BASKET1, 0)
        
        # Convert component positions to basket-equivalent units
        croissants_equivalent = positions.get(Product.CROISSANTS, 0) / BASKET_WEIGHTS[Product.CROISSANTS]
        jams_equivalent = positions.get(Product.JAMS, 0) / BASKET_WEIGHTS[Product.JAMS]
        djembes_equivalent = positions.get(Product.DJEMBES, 0) / BASKET_WEIGHTS[Product.DJEMBES]
        
        # Negative sign because short components = long spread
        basket_equivalent_component_position = -(croissants_equivalent + jams_equivalent + djembes_equivalent) / 3
        
        # Combine with basket position
        effective_position = basket_position + basket_equivalent_component_position
        
        return effective_position

    def execute_spread_trade(
        self,
        buy_basket: bool,
        quantity: int,
        order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        """
        Execute a spread trade by buying/selling the basket and the opposite for components
        
        Parameters:
        - buy_basket: True to buy basket (sell components), False to sell basket (buy components)
        - quantity: Number of spread units to trade
        - order_depths: Order depths for all products
        
        Returns:
        - Dictionary of orders for each product
        """
        result = {
            Product.PICNIC_BASKET1: [],
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: [],
        }
        
        # Limit quantity based on order book depth
        quantity = min(quantity, 10)  # Avoid trading too aggressively
        
        if quantity <= 0:
            return result
        
        # Basket orders
        basket_depth = order_depths[Product.PICNIC_BASKET1]
        
        if buy_basket:
            # Buy basket
            if basket_depth.sell_orders:
                best_ask = min(basket_depth.sell_orders.keys())
                result[Product.PICNIC_BASKET1].append(
                    Order(Product.PICNIC_BASKET1, best_ask, quantity)
                )
                
                # Sell components
                if order_depths[Product.CROISSANTS].buy_orders:
                    croissants_best_bid = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                    result[Product.CROISSANTS].append(
                        Order(Product.CROISSANTS, croissants_best_bid, -quantity * BASKET_WEIGHTS[Product.CROISSANTS])
                    )
                
                if order_depths[Product.JAMS].buy_orders:
                    jams_best_bid = max(order_depths[Product.JAMS].buy_orders.keys())
                    result[Product.JAMS].append(
                        Order(Product.JAMS, jams_best_bid, -quantity * BASKET_WEIGHTS[Product.JAMS])
                    )
                
                if order_depths[Product.DJEMBES].buy_orders:
                    djembes_best_bid = max(order_depths[Product.DJEMBES].buy_orders.keys())
                    result[Product.DJEMBES].append(
                        Order(Product.DJEMBES, djembes_best_bid, -quantity * BASKET_WEIGHTS[Product.DJEMBES])
                    )
        else:
            # Sell basket
            if basket_depth.buy_orders:
                best_bid = max(basket_depth.buy_orders.keys())
                result[Product.PICNIC_BASKET1].append(
                    Order(Product.PICNIC_BASKET1, best_bid, -quantity)
                )
                
                # Buy components
                if order_depths[Product.CROISSANTS].sell_orders:
                    croissants_best_ask = min(order_depths[Product.CROISSANTS].sell_orders.keys())
                    result[Product.CROISSANTS].append(
                        Order(Product.CROISSANTS, croissants_best_ask, quantity * BASKET_WEIGHTS[Product.CROISSANTS])
                    )
                
                if order_depths[Product.JAMS].sell_orders:
                    jams_best_ask = min(order_depths[Product.JAMS].sell_orders.keys())
                    result[Product.JAMS].append(
                        Order(Product.JAMS, jams_best_ask, quantity * BASKET_WEIGHTS[Product.JAMS])
                    )
                
                if order_depths[Product.DJEMBES].sell_orders:
                    djembes_best_ask = min(order_depths[Product.DJEMBES].sell_orders.keys())
                    result[Product.DJEMBES].append(
                        Order(Product.DJEMBES, djembes_best_ask, quantity * BASKET_WEIGHTS[Product.DJEMBES])
                    )
        
        return result

    def process_spread_strategy(
        self,
        order_depths: Dict[str, OrderDepth],
        positions: Dict[str, int],
        trader_data: Dict[str, Any]
    ) -> Dict[str, List[Order]]:
        """
        Process the adaptive spread trading strategy
        """
        # Initialize trader data structure if needed
        if Product.SPREAD not in trader_data:
            trader_data[Product.SPREAD] = {
                "spread_history": [],
                "mean_value": self.params[Product.SPREAD]["default_spread_mean"],
                "zscore_history": [],
                "position_history": [],
                "prev_zscore": 0,
            }
        
        spread_data = trader_data[Product.SPREAD]
        
        # Calculate current spread
        spread = self.calculate_spread(order_depths)
        if spread is None:
            return {}  # Not enough data to calculate spread
        
        # Track spread history
        spread_data["spread_history"].append(spread)
        
        # Use the default mean from parameters
        mean_value = self.params[Product.SPREAD]["default_spread_mean"]
        spread_data["mean_value"] = mean_value
        
        # Calculate z-score
        zscore = self.calculate_zscore(
            spread, 
            spread_data["spread_history"], 
            mean_value
        )
        
        spread_data["zscore_history"].append(zscore)
        
        # Calculate effective position
        effective_position = self.calculate_effective_position(positions)
        spread_data["position_history"].append(effective_position)
        
        print(f"Spread: {spread:.2f}, Z-Score: {zscore:.2f}, Position: {effective_position:.2f}")
        
        # Trading logic based on z-score
        result = {}
        
        # Buy spread (buy basket, sell components) when z-score < lower_threshold
        if zscore < self.params[Product.SPREAD]["zscore_threshold_lower"] and effective_position <= 0:
            print(f"BUY SIGNAL: z-score {zscore:.2f} < {self.params[Product.SPREAD]['zscore_threshold_lower']}")
            orders = self.execute_spread_trade(
                buy_basket=True,
                quantity=self.params[Product.SPREAD]["target_position"] - effective_position,
                order_depths=order_depths
            )
            result = orders
        
        # Sell spread (sell basket, buy components) when z-score > upper_threshold
        elif zscore > self.params[Product.SPREAD]["zscore_threshold_upper"] and effective_position >= 0:
            print(f"SELL SIGNAL: z-score {zscore:.2f} > {self.params[Product.SPREAD]['zscore_threshold_upper']}")
            orders = self.execute_spread_trade(
                buy_basket=False,
                quantity=effective_position + self.params[Product.SPREAD]["target_position"],
                order_depths=order_depths
            )
            result = orders
        
        # Exit long spread position when z-score crosses back above lower_threshold + buffer
        elif (effective_position > 0 and 
              zscore > self.params[Product.SPREAD]["zscore_threshold_lower"] + self.params[Product.SPREAD]["exit_buffer"]):
            print(f"EXIT LONG: z-score {zscore:.2f} > {self.params[Product.SPREAD]['zscore_threshold_lower'] + self.params[Product.SPREAD]['exit_buffer']}")
            orders = self.execute_spread_trade(
                buy_basket=False,
                quantity=effective_position,
                order_depths=order_depths
            )
            result = orders
        
        # Exit short spread position when z-score crosses back below upper_threshold - buffer
        elif (effective_position < 0 and 
              zscore < self.params[Product.SPREAD]["zscore_threshold_upper"] - self.params[Product.SPREAD]["exit_buffer"]):
            print(f"EXIT SHORT: z-score {zscore:.2f} < {self.params[Product.SPREAD]['zscore_threshold_upper'] - self.params[Product.SPREAD]['exit_buffer']}")
            orders = self.execute_spread_trade(
                buy_basket=True,
                quantity=-effective_position,
                order_depths=order_depths
            )
            result = orders
        
        # Store previous z-score for next iteration
        spread_data["prev_zscore"] = zscore
        
        return result

    def run(self, state: TradingState):
        """
        Main method called by the exchange. Takes a state object and returns a tuple of (orders, conversions, trader_data)
        """
        # Decode trader data
        trader_data = {}
        if state.traderData != None and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        
        # Initialize result
        result = {}
        
        # No conversions needed for basket trading
        conversions = 0
        
        # Check if all required products are available
        basket_products = [Product.PICNIC_BASKET1, Product.CROISSANTS, Product.JAMS, Product.DJEMBES]
        missing_products = [p for p in basket_products if p not in state.order_depths]
        
        if missing_products:
            print(f"Missing products: {missing_products}. Cannot execute strategy.")
            return result, conversions, jsonpickle.encode(trader_data)
        

        
        # Get current positions
        positions = {
            product: state.position.get(product, 0) 
            for product in basket_products
        }
        
        # Run the spread strategy
        spread_orders = self.process_spread_strategy(
            state.order_depths,
            positions,
            trader_data
        )
        
        # Merge spread orders into result
        for product, orders in spread_orders.items():
            if orders:
                result[product] = orders
        
        # Encode trader data for next round
        trader_data_encoded = jsonpickle.encode(trader_data)
        
        return result, conversions, trader_data_encoded