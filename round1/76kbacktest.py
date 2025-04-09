import json
from typing import Any, Dict, List
import math
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import collections


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self):
        # Configuration parameters
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
        
        
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Enhanced trading strategy with linear regression model for KELP
        """
        result = {}
        timestamp = state.timestamp
        
        # Initialize or load stored data
        
        # Update timestamps
        for symbol in state.order_depths:
            if symbol not in self.timestamps:
                self.timestamps[symbol] = []
            
            self.timestamps[symbol].append(timestamp)
            # Keep only recent timestamps
            if len(self.timestamps[symbol]) > 100:
                self.timestamps[symbol] = self.timestamps[symbol][-100:]
        
        #Process RAINFOREST_RESIN with improved strategy
        if "RAINFOREST_RESIN" in state.order_depths:
            self.process_rainforest_resin_simple(state, result)
        
        # Process KELP with our linear regression model
        if "KELP" in state.order_depths:
            # Use our new regression model instead of the simple approach
            self.process_kelp_regression(state, result)
        
        # No conversions in tutorial round
        conversions = 0
        
        # Store data for next round
        trader_data = json.dumps({
            "price_history": self.price_history,
            "volatility_estimates": self.volatility_estimates,
            "last_mid_prices": self.last_mid_prices,
            "timestamps": self.timestamps,
            "fair_value_estimates": self.fair_value_estimates,
            "realized_pnl": self.realized_pnl,
            "trades_completed": self.trades_completed,
        })
        
        # Return results
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
        
    def process_rainforest_resin_simple(self, state: TradingState, result: dict) -> None:
        symbol = "RAINFOREST_RESIN"
        order_depth = state.order_depths[symbol]
        current_position = state.position.get(symbol, 0)
    
        # Initialize orders list
        resin_orders = []
    
        # Define our fair value - Resin is stable around 10,000
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
                    logger.print(f"RESIN OPPORTUNITY BUY: {order_size} @ {price}")
    
        # Calculate undercut prices (to improve on best current prices)
        undercut_buy = best_buy_price + 1  # One tick better than current best buy
        undercut_sell = best_sell_price - 1  # One tick better than current best sell
    
        # Calculate our bid and ask prices
        bid_price = min(undercut_buy, fair_value - 1)  # Don't bid above fair value - 1
        ask_price = max(undercut_sell, fair_value + 1)  # Don't ask below fair value + 1
    
        # STEP 2: Add extra buying pressure if we're short
        if position < self.position_limit and current_position < 0:
            extra_buy_size = min(40, self.position_limit - position)
            extra_buy_price = min(undercut_buy + 1, fair_value - 1)
            resin_orders.append(Order(symbol, extra_buy_price, extra_buy_size))
            position += extra_buy_size
    
        # STEP 3: Add cautious buying when we're very long
        if position < self.position_limit and current_position > 15:
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
                    logger.print(f"RESIN OPPORTUNITY SELL: {-order_size} @ {price}")
    
        # STEP 6: Add extra selling pressure if we're long
        if position > -self.position_limit and current_position > 0:
            extra_sell_size = max(-40, -self.position_limit - position)
            extra_sell_price = max(undercut_sell - 1, fair_value + 1)
            resin_orders.append(Order(symbol, extra_sell_price, extra_sell_size))
            position += extra_sell_size
    
        # STEP 7: Add cautious selling when we're very short
        if position > -self.position_limit and current_position < -15:
            cautious_sell_size = max(-40, -self.position_limit - position)
            cautious_sell_price = max(undercut_sell + 1, fair_value + 1)  # Slightly better price
            resin_orders.append(Order(symbol, cautious_sell_price, cautious_sell_size))
            position += cautious_sell_size
    
        # STEP 8: Add regular ask to maintain market making presence
        if position > -self.position_limit:
            regular_sell_size = max(-40, -self.position_limit - position)
            resin_orders.append(Order(symbol, ask_price, regular_sell_size))
            position += regular_sell_size
    
        # Log strategy information
        logger.print(f"===== {symbol} Pearl-style Market Making =====")
        logger.print(f"Fair Value: {fair_value}")
        logger.print(f"Market - Best Bid: {best_buy_price}, Best Ask: {best_sell_price}, Spread: {best_sell_price-best_buy_price}")
        logger.print(f"Our Orders - Bid: {bid_price}, Ask: {ask_price}, Spread: {ask_price-bid_price}")
        logger.print(f"Position: {current_position}/{self.position_limit}")
        logger.print(f"Orders: {len(resin_orders)}")
    
        # Add orders to result
        result[symbol] = resin_orders

    def process_kelp_regression(self, state: TradingState, result: dict) -> None:
        """
        Process KELP using linear regression model similar to the BANANAS strategy
        """
        symbol = "KELP"
        order_depth = state.order_depths[symbol]
        current_position = state.position.get(symbol, 0)
        timestamp = state.timestamp
        position_limit = 50  # Position limit for KELP

        kelp_orders = []


        # Sort order books for easier processing
        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        best_sell_price = min(sell_orders.keys()) if sell_orders else None
        best_buy_price = max(buy_orders.keys()) if buy_orders else None

        # mid price
        if best_sell_price and best_buy_price:
            mid_price = (best_sell_price + best_buy_price) / 2
        elif best_sell_price:
            mid_price = best_sell_price
        elif best_buy_price:
            mid_price = best_buy_price
        else:
            # No orders on either side, use last known price or default
            mid_price = self.last_mid_prices.get(symbol, 2000)  # Default for KELP

        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(mid_price)

        # Keep only recent history for regression
        kelp_dim = 4
        if len(self.price_history[symbol]) > kelp_dim + 10:  # Keep extra for other calculations
            self.price_history[symbol] = self.price_history[symbol][-(kelp_dim + 10):]

        # Only predict if we have enough data points
        if len(self.price_history[symbol]) >= kelp_dim:
            next_price = self.predict_next_kelp_price()
            logger.print(f"KELP regression prediction: {next_price:.2f}")
        else:
            # Not enough data, use current mid price
            next_price = mid_price
            logger.print(f"KELP insufficient data for regression, using mid price: {next_price:.2f}")

        # Set acceptable price bounds
        price_lb = next_price - 1  # Lower bound
        price_ub = next_price + 1  # Upper bound

        # === Step 3: Market-take mispriced orders ===
        position = current_position

        # Buy underpriced asks
        for price, volume in sell_orders.items():
            if ((price <= price_lb) or (current_position < 0 and price == price_lb + 1)) and position < position_limit:
                # Order is underpriced or we're short and want to cover
                order_size = min(-volume, position_limit - position)
                if order_size > 0:
                    kelp_orders.append(Order(symbol, price, order_size))
                    position += order_size
                    logger.print(f"KELP OPPORTUNITY BUY: {order_size} @ {price}, Predicted: {next_price:.2f}")

        # Add limit buy order at our predicted price
        undercut_buy = best_buy_price + 1 if best_buy_price else price_lb
        bid_price = int(min(undercut_buy, price_lb))  # Don't bid above our lower bound

        if position < position_limit:
            buy_size = position_limit - position
            kelp_orders.append(Order(symbol, bid_price, buy_size))
            position += buy_size

        # Reset position tracking for sell orders
        position = current_position

        # Sell overpriced bids
        for price, volume in buy_orders.items():
            if ((price >= price_ub) or (current_position > 0 and price + 1 == price_ub)) and position > -position_limit:
                # Order is overpriced or we're long and want to reduce
                order_size = max(-volume, -position_limit - position)
                if order_size < 0:
                    kelp_orders.append(Order(symbol, price, order_size))
                    position += order_size
                    logger.print(f"KELP OPPORTUNITY SELL: {-order_size} @ {price}, Predicted: {next_price:.2f}")

        # Add limit sell order at our predicted price
        undercut_sell = best_sell_price - 1 if best_sell_price else price_ub
        ask_price = int(max(undercut_sell, price_ub))  # Don't ask below our upper bound

        if position > -position_limit:
            sell_size = -position_limit - position
            kelp_orders.append(Order(symbol, ask_price, sell_size))
            position += sell_size

        # Log strategy information
        logger.print(f"===== {symbol} Regression-based Trading =====")
        logger.print(f"Current Mid Price: {mid_price:.2f}, Predicted Next: {next_price:.2f}")
        logger.print(f"Price Bounds - Lower: {price_lb}, Upper: {price_ub}")
        logger.print(f"Market - Best Bid: {best_buy_price}, Best Ask: {best_sell_price}")
        logger.print(f"Our Orders - Bid: {bid_price}, Ask: {ask_price}")
        logger.print(f"Position: {current_position}/{position_limit}")
        logger.print(f"Orders: {len(kelp_orders)}")

        # Store current mid price for future reference
        self.last_mid_prices[symbol] = mid_price

        # Add orders to result
        result[symbol] = kelp_orders

    def predict_next_kelp_price(self):
        """
        Predict the next KELP price using the calibrated linear regression model
        """
        symbol = "KELP"

        # Check if we have enough data
        if symbol not in self.price_history or len(self.price_history[symbol]) < 4:
            # Not enough data, return last known price or default
            return self.last_mid_prices.get(symbol, 2000)

        # Get the most recent 4 prices (most recent first)
        recent_prices = self.price_history[symbol][-4:]

        # KELP-specific coefficients based on full dataset analysis
        coef = [0.335230, 0.260282, 0.204951, 0.198536]
        intercept = 2.024153

        # linear regression formula
        next_price = intercept
        for i, price in enumerate(recent_prices):
            next_price += price * coef[i]

        return next_price