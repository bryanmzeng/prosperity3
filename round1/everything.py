import json
from typing import Any, Dict, List
import math
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import collections
import statistics
import numpy as np
import pandas as pd


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
                    self.compress_state(state, self.truncate(
                        state.traderData, max_item_length)),
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
            compressed.append(
                [listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [
                order_depth.buy_orders, order_depth.sell_orders]

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
        self.target_inventory = 0  # Target inventory level (default: neutral)

        # State variables
        self.price_history = {}
        self.volatility_estimates = {}
        self.last_mid_prices = {}
        self.timestamps = {}
        self.fair_value_estimates = {}
        self.realized_pnl = {}
        self.trades_completed = {}

        # KELP specific parameters (keeping these for compatibility)
        self.kelp_dim = 4  # Number of lags for KELP regression
        self.kelp_cache = []
        self.last_mid_prices = {}  # To store last known mid prices

        # SQUID_INK specific parameters
        self.squid_ink_features = []
        self.squid_ink_trades = []  # Recent trade data for SQUID_INK
        self.squid_ink_windows = [3, 5, 10]
        self.squid_ink_lags = [1, 2, 3]
        
        # Arbitrage parameters
        self.kelp_squid_diffs = []
        self.expected_kelp_squid_diff = 3  # Default based on the conversation hint

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Enhanced trading strategy with ensemble model for SQUID_INK and arbitrage opportunities
        """
        result = {}
        timestamp = state.timestamp

        # Update timestamps
        for symbol in state.order_depths:
            if symbol not in self.timestamps:
                self.timestamps[symbol] = []

            self.timestamps[symbol].append(timestamp)
            # Keep only recent timestamps
            if len(self.timestamps[symbol]) > 100:
                self.timestamps[symbol] = self.timestamps[symbol][-100:]

        # First, check for and execute arbitrage opportunities
        # This takes precedence over other strategies
        if "KELP" in state.order_depths and "SQUID_INK" in state.order_depths:
            self.execute_arbitrage(state, result)

        # Process RAINFOREST_RESIN with improved strategy
        if "RAINFOREST_RESIN" in state.order_depths:
            self.process_rainforest_resin_simple(state, result)

        # Process KELP with our linear regression model
        if "KELP" in state.order_depths:
            # Use our new regression model
            self.process_kelp_regression(state, result)

        # Process SQUID_INK with enhanced ensemble ML-based strategy
        if "SQUID_INK" in state.order_depths:
            # Analyze the order book without market maker to see true market dynamics
            filtered_buys, filtered_sells = self.analyze_order_book_without_mm(state, "SQUID_INK")
            # We could use these filtered books for additional insights
            
            # Execute our enhanced SQUID_INK strategy
            self.process_squid_ink_enhanced(state, result)

        # No conversions in tutorial round
        
        # Track recent market trades for SQUID_INK
        if "SQUID_INK" in state.market_trades:
            self.squid_ink_trades.extend(state.market_trades["SQUID_INK"])
            if len(self.squid_ink_trades) > 100:
                self.squid_ink_trades = self.squid_ink_trades[-100:]
        
        # Also track KELP trades if available for correlation analysis
        if "KELP" in state.market_trades:
            if not hasattr(self, 'kelp_trades'):
                self.kelp_trades = []
            self.kelp_trades.extend(state.market_trades["KELP"])
            if len(self.kelp_trades) > 100:
                self.kelp_trades = self.kelp_trades[-100:]
        
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
            "squid_ink_features": self.squid_ink_features,
            "kelp_squid_diffs": self.kelp_squid_diffs,
            "expected_kelp_squid_diff": self.expected_kelp_squid_diff
        })

        # Return results
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    
    def process_rainforest_resin_simple(self, state: TradingState, result: dict) -> None:
        symbol = "RAINFOREST_RESIN"
        order_depth = state.order_depths[symbol]
        current_position = state.position.get(symbol, 0)
        resin_orders = []

        fair_value = 10000

        # Sort order books for easier processing
        sell_orders = collections.OrderedDict(
            sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(
            sorted(order_depth.buy_orders.items(), reverse=True))

        # Extract best prices
        best_sell_price = min(
            sell_orders.keys()) if sell_orders else fair_value + 5
        best_buy_price = max(
            buy_orders.keys()) if buy_orders else fair_value - 5

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
                    logger.print(
                        f"RESIN OPPORTUNITY BUY: {order_size} @ {price}")

        # Calculate undercut prices (to improve on best current prices)
        undercut_buy = best_buy_price + 1  # One tick better than current best buy
        undercut_sell = best_sell_price - 1  # One tick better than current best sell

        # Calculate our bid and ask prices
        # Don't bid above fair value - 1
        bid_price = min(undercut_buy, fair_value - 2)
        # Don't ask below fair value + 1
        ask_price = max(undercut_sell, fair_value + 2)

        # STEP 2: Add extra buying pressure if we're short
        if position < self.position_limit and current_position < -10:
            extra_buy_size = min(45, self.position_limit - position)
            extra_buy_price = min(undercut_buy + 1, fair_value - 1)
            resin_orders.append(Order(symbol, extra_buy_price, extra_buy_size))
            position += extra_buy_size

        # STEP 3: Add cautious buying when we're very long
        if position < self.position_limit and current_position > 35:
            cautious_buy_size = min(40, self.position_limit - position)
            # Slightly worse price
            cautious_buy_price = min(undercut_buy - 1, fair_value - 1)
            resin_orders.append(
                Order(symbol, cautious_buy_price, cautious_buy_size))
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
                    logger.print(
                        f"RESIN OPPORTUNITY SELL: {-order_size} @ {price}")

        # STEP 6: Add extra selling pressure if we're long
        if position > -self.position_limit and current_position > 35:
            extra_sell_size = max(-40, -self.position_limit - position)
            extra_sell_price = max(undercut_sell - 1, fair_value + 1)
            resin_orders.append(
                Order(symbol, extra_sell_price, extra_sell_size))
            position += extra_sell_size

        # STEP 7: Add cautious selling when we're very short
        if position > -self.position_limit and current_position < -35:
            cautious_sell_size = max(-40, -self.position_limit - position)
            # Slightly better price
            cautious_sell_price = max(undercut_sell + 1, fair_value + 1)
            resin_orders.append(
                Order(symbol, cautious_sell_price, cautious_sell_size))
            position += cautious_sell_size

        # STEP 8: Add regular ask to maintain market making presence
        if position > -self.position_limit:
            regular_sell_size = max(-40, -self.position_limit - position)
            resin_orders.append(Order(symbol, ask_price, regular_sell_size))
            position += regular_sell_size

        # Log strategy information
        logger.print(f"===== {symbol} Pearl-style Market Making =====")
        logger.print(f"Fair Value: {fair_value}")
        logger.print(
            f"Market - Best Bid: {best_buy_price}, Best Ask: {best_sell_price}, Spread: {best_sell_price-best_buy_price}")
        logger.print(
            f"Our Orders - Bid: {bid_price}, Ask: {ask_price}, Spread: {ask_price-bid_price}")
        logger.print(f"Position: {current_position}/{self.position_limit}")
        logger.print(f"Orders: {len(resin_orders)}")

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
        sell_orders = collections.OrderedDict(
            sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(
            sorted(order_depth.buy_orders.items(), reverse=True))

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

        # Log strategy information
        logger.print(f"===== {symbol} Regression-based Trading =====")
        logger.print(
            f"Current Mid Price: {mid_price:.2f}, Predicted Next: {next_price:.2f}")
        logger.print(f"Price Bounds - Lower: {acc_bid}, Upper: {acc_ask}")
        logger.print(
            f"Market - Best Bid: {best_buy_price}, Best Ask: {best_sell_price}")
        logger.print(
            f"Position: {state.position.get(symbol, 0)}/{position_limit}")
        logger.print(f"Orders: {len(kelp_orders)}")

        # store current mid price for future reference
        self.last_mid_prices[symbol] = mid_price

        # add orders to result
        result[symbol] = kelp_orders

    def process_squid_ink_ensemble(self, state: TradingState, result: dict) -> None:
        """
        Process SQUID_INK using ensemble model from squid_ink.py
        """
        symbol = "SQUID_INK"
        position_limit = 50
        squid_orders = []

        # Get current order book and position
        order_depth = state.order_depths[symbol]
        current_position = state.position.get(symbol, 0)

        # Sort order books for easier processing
        sell_orders = collections.OrderedDict(
            sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(
            sorted(order_depth.buy_orders.items(), reverse=True))

        # Calculate mid price and update history
        best_sell_price = min(sell_orders.keys()) if sell_orders else None
        best_buy_price = max(buy_orders.keys()) if buy_orders else None

        if best_sell_price and best_buy_price:
            mid_price = (best_sell_price + best_buy_price) / 2
        elif best_sell_price:
            mid_price = best_sell_price
        elif best_buy_price:
            mid_price = best_buy_price
        else:
            mid_price = self.last_mid_prices.get(
                symbol, 5000)  # Default if no prices available

        # Update price history for SQUID_INK
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(mid_price)
        # Limit history size to prevent memory issues
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]

        # Create feature record
        self.update_squid_ink_features(state, symbol, mid_price)

        # Get predicted price from ensemble model
        # When we don't have enough data, use a simple spread-based approach
        if len(self.squid_ink_features) < 20:  # Need enough data for features
            # Simple spread-based fair value
            fair_value = mid_price
            predicted_diff = 0
        else:
            # Use ensemble model prediction
            predicted_price, predicted_diff = self.predict_with_ensemble(
                mid_price)
            fair_value = predicted_price

        # Dynamic threshold based on prediction confidence
        threshold = max(5, abs(predicted_diff))

        # Calculate acceptable bid/ask prices based on prediction
        acc_bid = int(fair_value - threshold)
        acc_ask = int(fair_value + threshold)

        # Current position
        position = current_position

        # STEP 1: Take advantage of mispriced orders first based on our prediction
        for price, volume in sell_orders.items():
            if price < acc_bid and position < position_limit:
                order_size = min(-volume, position_limit - position)
                if order_size > 0:
                    squid_orders.append(Order(symbol, price, order_size))
                    position += order_size
                    logger.print(
                        f"SQUID OPPORTUNITY BUY: {order_size} @ {price}")

        # Calculate undercut prices
        undercut_buy = best_buy_price + 1 if best_buy_price else acc_bid - 1
        undercut_sell = best_sell_price - 1 if best_sell_price else acc_ask + 1

        # Place bids and asks based on our prediction
        bid_price = min(undercut_buy, acc_bid)
        ask_price = max(undercut_sell, acc_ask)

        # STEP 2: Add our bids - make bid size dependent on prediction strength
        if predicted_diff > 1:  # Positive momentum - be more aggressive buying
            bid_size = min(40, position_limit - position)
            if bid_size > 0 and position < position_limit:
                squid_orders.append(Order(symbol, bid_price, bid_size))
                position += bid_size
        else:
            bid_size = min(25, position_limit - position)
            if bid_size > 0 and position < position_limit:
                squid_orders.append(Order(symbol, bid_price, bid_size))
                position += bid_size

        # Reset for sell orders
        position = current_position

        # STEP 3: Sell overpriced bids
        for price, volume in buy_orders.items():
            if price > acc_ask and position > -position_limit:
                order_size = max(-volume, -position_limit - position)
                if order_size < 0:  # Must be negative for sell orders
                    squid_orders.append(Order(symbol, price, order_size))
                    position += order_size
                    logger.print(
                        f"SQUID OPPORTUNITY SELL: {-order_size} @ {price}")

        # STEP 4: Add our asks - make ask size dependent on prediction strength
        if predicted_diff < -1:  # Negative momentum - be more aggressive selling
            ask_size = max(-40, -position_limit - position)
            if ask_size < 0 and position > -position_limit:
                squid_orders.append(Order(symbol, ask_price, ask_size))
                position += ask_size
        else:
            ask_size = max(-25, -position_limit - position)
            if ask_size < 0 and position > -position_limit:
                squid_orders.append(Order(symbol, ask_price, ask_size))
                position += ask_size

        # Log strategy information
        logger.print(f"===== {symbol} Ensemble Model Trading =====")
        logger.print(
            f"Current Price: {mid_price:.2f}, Fair Value: {fair_value:.2f}, Predicted Diff: {predicted_diff:.2f}")
        logger.print(f"Price Bounds - Bid: {acc_bid}, Ask: {acc_ask}")
        logger.print(
            f"Market - Best Bid: {best_buy_price}, Best Ask: {best_sell_price}")
        logger.print(f"Position: {current_position}/{position_limit}")
        logger.print(f"Orders: {len(squid_orders)}")

        # Store current mid price for reference
        self.last_mid_prices[symbol] = mid_price

        # Add orders to result
        result[symbol] = squid_orders

    def update_squid_ink_features(self, state: TradingState, symbol: str, mid_price: float) -> None:
        """
        Update features for SQUID_INK ensemble model
        """
        order_depth = state.order_depths[symbol]

        # Basic feature dictionary
        features = {
            "mid_price": mid_price,
            "timestamp": state.timestamp,
        }

        # Trade-based features
        recent_trades = self.squid_ink_trades[-20:]  # Recent N trades
        if recent_trades:
            buyer_count = sum(1 for t in recent_trades if t.buyer.startswith("SUBMISSION"))
            seller_count = sum(1 for t in recent_trades if t.seller.startswith("SUBMISSION"))
            total_trades = len(recent_trades)
            buy_ratio = buyer_count / total_trades if total_trades else 0
            sell_ratio = seller_count / total_trades if total_trades else 0
            avg_trade_price = sum(t.price for t in recent_trades) / total_trades
            avg_trade_size = sum(t.quantity for t in recent_trades) / total_trades

            features["trade_buy_ratio"] = buy_ratio
            features["trade_sell_ratio"] = sell_ratio
            features["avg_trade_price"] = avg_trade_price
            features["avg_trade_size"] = avg_trade_size
        else:
            features["trade_buy_ratio"] = 0
            features["trade_sell_ratio"] = 0
            features["avg_trade_price"] = mid_price
            features["avg_trade_size"] = 1

    # Add order book features
        if symbol in state.order_depths:
            sell_orders = order_depth.sell_orders
            buy_orders = order_depth.buy_orders

        # Calculate spread
            if sell_orders and buy_orders:
                best_ask = min(sell_orders.keys())
                best_bid = max(buy_orders.keys())
                features["spread"] = best_ask - best_bid
                features["ask_price_1"] = best_ask
                features["bid_price_1"] = best_bid

            # Calculate bid/ask volumes
                best_ask_vol = abs(sell_orders[best_ask])
                best_bid_vol = abs(buy_orders[best_bid])
                features["ask_volume_1"] = best_ask_vol
                features["bid_volume_1"] = best_bid_vol

            # Volume imbalance
                features["volume_imbalance"] = (
                best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol + 1e-6)

            # Try to get level 2 and 3 prices if available (approximate)
                sorted_asks = sorted(sell_orders.keys())
                sorted_bids = sorted(buy_orders.keys(), reverse=True)

                if len(sorted_asks) > 1:
                    features["ask_price_2"] = sorted_asks[1]
                    features["ask_volume_2"] = abs(sell_orders[sorted_asks[1]])
                else:
                    features["ask_price_2"] = best_ask + 5
                    features["ask_volume_2"] = 0

                if len(sorted_asks) > 2:
                    features["ask_price_3"] = sorted_asks[2]
                    features["ask_volume_3"] = abs(sell_orders[sorted_asks[2]])
                else:
                    features["ask_price_3"] = best_ask + 10
                    features["ask_volume_3"] = 0

                if len(sorted_bids) > 1:
                    features["bid_price_2"] = sorted_bids[1]
                    features["bid_volume_2"] = abs(buy_orders[sorted_bids[1]])
                else:
                    features["bid_price_2"] = best_bid - 5
                    features["bid_volume_2"] = 0

                if len(sorted_bids) > 2:
                    features["bid_price_3"] = sorted_bids[2]
                    features["bid_volume_3"] = abs(buy_orders[sorted_bids[2]])
                else:
                    features["bid_price_3"] = best_bid - 10
                    features["bid_volume_3"] = 0

            # Calculate slopes
                if features["ask_volume_1"] != features["ask_volume_3"]:
                    features["ask_slope"] = (features["ask_price_3"] - features["ask_price_1"]) / (
                        features["ask_volume_3"] - features["ask_volume_1"] + 1e-6)
                else:
                    features["ask_slope"] = 0

                if features["bid_volume_1"] != features["bid_volume_3"]:
                    features["bid_slope"] = (features["bid_price_1"] - features["bid_price_3"]) / (
                        features["bid_volume_1"] - features["bid_volume_3"] + 1e-6)
                else:
                    features["bid_slope"] = 0

    # Add lag values
        price_history = self.price_history.get(symbol, [])
        for lag in self.squid_ink_lags:
            if len(price_history) > lag:
                features[f"lag_{lag}"] = price_history[-lag - 1]
            else:
                features[f"lag_{lag}"] = mid_price

    # Add rolling window features
        for window in self.squid_ink_windows:
            if len(price_history) >= window:
                window_data = price_history[-window:]
                features[f"roll_mean_{window}"] = sum(window_data) / window

                if len(window_data) > 1:
                # Calculate standard deviation
                    features[f"roll_std_{window}"] = np.std(window_data)

                # Calculate return features
                    if len(price_history) > window + 1:
                        returns = []
                        for i in range(len(window_data) - 1):
                            if window_data[i] != 0:  # Avoid division by zero
                                returns.append(
                                    (window_data[i + 1] - window_data[i]) / window_data[i])
                        features[f"return_std_{window}"] = np.std(returns) if returns else 0
            else:
                features[f"roll_mean_{window}"] = mid_price
                features[f"roll_std_{window}"] = 0
                features[f"return_std_{window}"] = 0

    # Add time features
        second = state.timestamp % 86400
        minute = second // 60
        hour = minute // 60
        features["second"] = second
        features["minute"] = minute
        features["hour"] = hour
        features["sin_time"] = np.sin(2 * np.pi * minute / 1440)
        features["cos_time"] = np.cos(2 * np.pi * minute / 1440)

    # Add to features list
        self.squid_ink_features.append(features)
        if len(self.squid_ink_features) > 100:  # Keep last 100 records
            self.squid_ink_features = self.squid_ink_features[-100:]
        
    def detect_arbitrage_opportunities(self, state: TradingState) -> list:
        """
        Detect arbitrage opportunities between KELP and SQUID_INK
        Returns a list of arbitrage opportunities
        """
        arbitrage_opps = []
        
        # Check if both products exist in the order depths
        if "KELP" not in state.order_depths or "SQUID_INK" not in state.order_depths:
            return arbitrage_opps
        
        kelp_depth = state.order_depths["KELP"]
        squid_depth = state.order_depths["SQUID_INK"]
        
        # Sort order books
        kelp_sells = collections.OrderedDict(sorted(kelp_depth.sell_orders.items()))
        kelp_buys = collections.OrderedDict(sorted(kelp_depth.buy_orders.items(), reverse=True))
        squid_sells = collections.OrderedDict(sorted(squid_depth.sell_orders.items()))
        squid_buys = collections.OrderedDict(sorted(squid_depth.buy_orders.items(), reverse=True))
        
        # Extract best prices
        best_kelp_sell = min(kelp_sells.keys()) if kelp_sells else float('inf')
        best_kelp_buy = max(kelp_buys.keys()) if kelp_buys else -1
        best_squid_sell = min(squid_sells.keys()) if squid_sells else float('inf')
        best_squid_buy = max(squid_buys.keys()) if squid_buys else -1
        
        # Calculate the fair price relationship between KELP and SQUID_INK
        # Based on the conversation, there appears to be a correlation where SQUID_INK = KELP + 3
        expected_diff = getattr(self, 'expected_kelp_squid_diff', 3)  # Default to 3 based on hints
        
        # Check for arbitrage opportunities
        
        # 1. KELP price is too high relative to SQUID_INK
        # Buy SQUID_INK, Sell KELP
        if best_squid_sell < float('inf') and best_kelp_buy > -1:
            if best_kelp_buy > (best_squid_sell - expected_diff + 1):  # +1 for profit threshold
                arb_profit = best_kelp_buy - (best_squid_sell - expected_diff)
                # Calculate max size based on available orders and position limits
                squid_buy_size = min(-squid_sells[best_squid_sell], 
                                     self.position_limit - state.position.get("SQUID_INK", 0))
                kelp_sell_size = min(kelp_buys[best_kelp_buy], 
                                    self.position_limit + state.position.get("KELP", 0))
                
                max_size = min(squid_buy_size, kelp_sell_size)
                
                if max_size > 0 and arb_profit > 0:
                    arbitrage_opps.append({
                        "type": "BUY_SQUID_SELL_KELP",
                        "kelp_price": best_kelp_buy,
                        "squid_price": best_squid_sell,
                        "size": max_size,
                        "profit": arb_profit * max_size
                    })
        
        # 2. SQUID_INK price is too high relative to KELP
        # Buy KELP, Sell SQUID_INK
        if best_kelp_sell < float('inf') and best_squid_buy > -1:
            if best_squid_buy > (best_kelp_sell + expected_diff + 1):  # +1 for profit threshold
                arb_profit = best_squid_buy - (best_kelp_sell + expected_diff)
                # Calculate max size based on available orders and position limits
                kelp_buy_size = min(-kelp_sells[best_kelp_sell], 
                                   self.position_limit - state.position.get("KELP", 0))
                squid_sell_size = min(squid_buys[best_squid_buy], 
                                     self.position_limit + state.position.get("SQUID_INK", 0))
                
                max_size = min(kelp_buy_size, squid_sell_size)
                
                if max_size > 0 and arb_profit > 0:
                    arbitrage_opps.append({
                        "type": "BUY_KELP_SELL_SQUID",
                        "kelp_price": best_kelp_sell,
                        "squid_price": best_squid_buy,
                        "size": max_size,
                        "profit": arb_profit * max_size
                    })
        
        # Sort opportunities by profit
        arbitrage_opps.sort(key=lambda x: x["profit"], reverse=True)
        return arbitrage_opps

    def execute_arbitrage(self, state: TradingState, result: dict) -> None:
        """
        Execute detected arbitrage opportunities between KELP and SQUID_INK
        """
        arbitrage_opps = self.detect_arbitrage_opportunities(state)
        
        if not arbitrage_opps:
            return
        
        for opp in arbitrage_opps:
            logger.print(f"===== EXECUTING ARBITRAGE OPPORTUNITY =====")
            logger.print(f"Type: {opp['type']}")
            logger.print(f"KELP Price: {opp['kelp_price']}, SQUID_INK Price: {opp['squid_price']}")
            logger.print(f"Size: {opp['size']}, Expected Profit: {opp['profit']}")
            
            if opp["type"] == "BUY_SQUID_SELL_KELP":
                # Buy SQUID_INK, Sell KELP
                size = opp["size"]
                
                # Create SQUID_INK buy order
                if "SQUID_INK" not in result:
                    result["SQUID_INK"] = []
                result["SQUID_INK"].append(Order("SQUID_INK", opp["squid_price"], size))
                
                # Create KELP sell order
                if "KELP" not in result:
                    result["KELP"] = []
                result["KELP"].append(Order("KELP", opp["kelp_price"], -size))
                
            elif opp["type"] == "BUY_KELP_SELL_SQUID":
                # Buy KELP, Sell SQUID_INK
                size = opp["size"]
                
                # Create KELP buy order
                if "KELP" not in result:
                    result["KELP"] = []
                result["KELP"].append(Order("KELP", opp["kelp_price"], size))
                
                # Create SQUID_INK sell order
                if "SQUID_INK" not in result:
                    result["SQUID_INK"] = []
                result["SQUID_INK"].append(Order("SQUID_INK", opp["squid_price"], -size))
            
            # After executing one opportunity, we update our understanding of the price relationship
            # This helps adjust our arbitrage model over time
            self.update_correlation_model(opp)

    def update_correlation_model(self, opportunity):
        """
        Update our understanding of the KELP-SQUID_INK price relationship
        """
        # Record the observed price difference
        observed_diff = opportunity["squid_price"] - opportunity["kelp_price"]
        
        # Initialize if not already there
        if not hasattr(self, 'kelp_squid_diffs'):
            self.kelp_squid_diffs = []
            
        self.kelp_squid_diffs.append(observed_diff)
        
        # Keep only recent observations
        if len(self.kelp_squid_diffs) > 100:
            self.kelp_squid_diffs = self.kelp_squid_diffs[-100:]
        
        # Update our expected difference - use a weighted average with recent values weighted more heavily
        if len(self.kelp_squid_diffs) >= 10:
            # Use exponential weighting to favor recent observations
            weights = [0.5 ** (len(self.kelp_squid_diffs) - i) for i in range(len(self.kelp_squid_diffs))]
            total_weight = sum(weights)
            weighted_diffs = sum(w * d for w, d in zip(weights, self.kelp_squid_diffs)) / total_weight
            
            # Update the expected difference
            self.expected_kelp_squid_diff = weighted_diffs
        
        # Log the update
        if hasattr(self, 'expected_kelp_squid_diff'):
            logger.print(f"Updated KELP-SQUID correlation model: expected diff = {self.expected_kelp_squid_diff:.2f}")

    def analyze_order_book_without_mm(self, state: TradingState, symbol: str):
        """
        Analyze the order book without considering market maker orders
        to find the true market participants as suggested in the conversation
        """
        if symbol not in state.order_depths:
            return None, None
        
        order_depth = state.order_depths[symbol]
        
        # Identify potential market maker patterns
        # Market makers often place orders at specific price levels or with specific sizes
        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
        
        # Filter suspected market maker orders
        # This is a simple approach - in reality, you'd need more sophisticated detection
        filtered_sells = {}
        filtered_buys = {}
        
        # Assuming market maker places orders at round numbers or certain patterns
        # This is a placeholder - you would need to refine this based on observed patterns
        for price, volume in sell_orders.items():
            # Skip orders that match suspected market maker patterns
            if price % 5 == 0 and abs(volume) >= 20:  # Example pattern
                continue
            filtered_sells[price] = volume
        
        for price, volume in buy_orders.items():
            # Skip orders that match suspected market maker patterns
            if price % 5 == 0 and abs(volume) >= 20:  # Example pattern
                continue
            filtered_buys[price] = volume
        
        # Return the filtered order books
        return filtered_buys, filtered_sells
    
    def process_squid_ink_enhanced(self, state: TradingState, result: dict) -> None:
        """
        Enhanced SQUID_INK processing that checks for correlation with KELP
        """
        symbol = "SQUID_INK"
        position_limit = 50
        squid_orders = []

        # Get current order book and position
        order_depth = state.order_depths[symbol]
        current_position = state.position.get(symbol, 0)

        # Sort order books for easier processing
        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        # Calculate mid price and update history
        best_sell_price = min(sell_orders.keys()) if sell_orders else None
        best_buy_price = max(buy_orders.keys()) if buy_orders else None

        if best_sell_price and best_buy_price:
            mid_price = (best_sell_price + best_buy_price) / 2
        elif best_sell_price:
            mid_price = best_sell_price
        elif best_buy_price:
            mid_price = best_buy_price
        else:
            mid_price = self.last_mid_prices.get(symbol, 5000)  # Default if no prices available

        # Update price history for SQUID_INK
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append(mid_price)
        # Limit history size to prevent memory issues
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]

        # Create feature record
        self.update_squid_ink_features(state, symbol, mid_price)
        
        # Check correlation with KELP for additional insights
        kelp_mid_price = None
        if "KELP" in state.order_depths:
            kelp_depth = state.order_depths["KELP"]
            kelp_sells = collections.OrderedDict(sorted(kelp_depth.sell_orders.items()))
            kelp_buys = collections.OrderedDict(sorted(kelp_depth.buy_orders.items(), reverse=True))
            
            kelp_best_sell = min(kelp_sells.keys()) if kelp_sells else None
            kelp_best_buy = max(kelp_buys.keys()) if kelp_buys else None
            
            if kelp_best_sell and kelp_best_buy:
                kelp_mid_price = (kelp_best_sell + kelp_best_buy) / 2
            elif kelp_best_sell:
                kelp_mid_price = kelp_best_sell
            elif kelp_best_buy:
                kelp_mid_price = kelp_best_buy
        
        # Use correlation with KELP to refine our fair value estimate if possible
        if kelp_mid_price is not None:
            # Get the expected difference (default to 3 if not yet calculated)
            expected_diff = getattr(self, 'expected_kelp_squid_diff', 3)
            kelp_implied_squid_price = kelp_mid_price + expected_diff
            
            # Blend our ensemble prediction with the KELP-implied price
            ensemble_predicted_price, ensemble_diff = self.predict_with_ensemble(mid_price)
            correlation_weight = 0.3  # How much to weight the correlation vs the ensemble
            
            fair_value = (ensemble_predicted_price * (1 - correlation_weight) + 
                         kelp_implied_squid_price * correlation_weight)
            
            # Adjust predicted diff based on the correlation
            kelp_implied_diff = kelp_implied_squid_price - mid_price
            predicted_diff = (ensemble_diff * (1 - correlation_weight) + 
                            kelp_implied_diff * correlation_weight)
            
            logger.print(f"SQUID_INK Enhanced Valuation:")
            logger.print(f"  Ensemble Value: {ensemble_predicted_price:.2f}")
            logger.print(f"  KELP-Implied Value: {kelp_implied_squid_price:.2f}")
            logger.print(f"  Blended Fair Value: {fair_value:.2f}")
        else:
            # Fall back to ensemble prediction if KELP data isn't available
            fair_value, predicted_diff = self.predict_with_ensemble(mid_price)

        # Dynamic threshold based on prediction confidence
        threshold = max(2, abs(predicted_diff))

        # Calculate acceptable bid/ask prices based on prediction
        acc_bid = int(fair_value - threshold)
        acc_ask = int(fair_value + threshold)

        # Current position
        position = current_position

        # STEP 1: Take advantage of mispriced orders first based on our prediction
        for price, volume in sell_orders.items():
            if price < acc_bid and position < position_limit:
                order_size = min(-volume, position_limit - position)
                if order_size > 0:
                    squid_orders.append(Order(symbol, price, order_size))
                    position += order_size
                    logger.print(f"SQUID OPPORTUNITY BUY: {order_size} @ {price}")

        # Calculate undercut prices
        undercut_buy = best_buy_price + 1 if best_buy_price else acc_bid - 1
        undercut_sell = best_sell_price - 1 if best_sell_price else acc_ask + 1

        # Place bids and asks based on our prediction
        bid_price = min(undercut_buy, acc_bid)
        ask_price = max(undercut_sell, acc_ask)

        # STEP 2: Add our bids - make bid size dependent on prediction strength
        if predicted_diff > 1:  # Positive momentum - be more aggressive buying
            bid_size = min(40, position_limit - position)
            if bid_size > 0 and position < position_limit:
                squid_orders.append(Order(symbol, bid_price, bid_size))
                position += bid_size
        else:
            bid_size = min(25, position_limit - position)
            if bid_size > 0 and position < position_limit:
                squid_orders.append(Order(symbol, bid_price, bid_size))
                position += bid_size

        # Reset for sell orders
        position = current_position

        # STEP 3: Sell overpriced bids
        for price, volume in buy_orders.items():
            if price > acc_ask and position > -position_limit:
                order_size = max(-volume, -position_limit - position)
                if order_size < 0:  # Must be negative for sell orders
                    squid_orders.append(Order(symbol, price, order_size))
                    position += order_size
                    logger.print(f"SQUID OPPORTUNITY SELL: {-order_size} @ {price}")

        # STEP 4: Add our asks - make ask size dependent on prediction strength
        if predicted_diff < -1:  # Negative momentum - be more aggressive selling
            ask_size = max(-40, -position_limit - position)
            if ask_size < 0 and position > -position_limit:
                squid_orders.append(Order(symbol, ask_price, ask_size))
                position += ask_size
        else:
            ask_size = max(-25, -position_limit - position)
            if ask_size < 0 and position > -position_limit:
                squid_orders.append(Order(symbol, ask_price, ask_size))
                position += ask_size

        # Log strategy information
        logger.print(f"===== {symbol} Enhanced Model Trading =====")
        logger.print(f"Current Price: {mid_price:.2f}, Fair Value: {fair_value:.2f}, Predicted Diff: {predicted_diff:.2f}")
        logger.print(f"Price Bounds - Bid: {acc_bid}, Ask: {acc_ask}")
        logger.print(f"Market - Best Bid: {best_buy_price}, Best Ask: {best_sell_price}")
        logger.print(f"Position: {current_position}/{position_limit}")
        logger.print(f"Orders: {len(squid_orders)}")

        # Store current mid price for reference
        self.last_mid_prices[symbol] = mid_price

        # Add orders to result
        result[symbol] = squid_orders
    
    def predict_with_ensemble(self, current_price: float) -> tuple[float, float]:
        """
        Use ensemble of simpler models to predict SQUID_INK price
        Returns (predicted_price, predicted_difference)
        """
        if len(self.squid_ink_features) < 3:
            return current_price, 0

        # Recent features
        recent = self.squid_ink_features[-3:]

        # 1. Moving average model (weights most recent prices higher)
        ma_weights = [0.2, 0.3, 0.5]  # Most recent has highest weight
        ma_prediction = sum(recent[i]["mid_price"] * ma_weights[i]
                            for i in range(len(recent)))

        # 2. Momentum model
        momentum = 0
        if len(recent) >= 2:
            price_diffs = [recent[i]["mid_price"] - recent[i-1]
                           ["mid_price"] for i in range(1, len(recent))]
            momentum = sum(price_diffs) / len(price_diffs)
        momentum_prediction = current_price + momentum * 2  # Project momentum forward

        # 3. Order imbalance model
        imbalance_prediction = current_price
        if "volume_imbalance" in recent[-1]:
            # Adjust based on order book imbalance
            imbalance = recent[-1]["volume_imbalance"]
            imbalance_factor = 5  # Scale factor for imbalance
            imbalance_prediction = current_price + imbalance * imbalance_factor

        # 4. Mean-reversion model
        reversion_prediction = current_price
        if len(self.price_history.get("SQUID_INK", [])) > 20:
            long_term_mean = sum(self.price_history["SQUID_INK"][-20:]) / 20
            # If price deviates too much from mean, predict reversion
            deviation = current_price - long_term_mean
            reversion_strength = 0.3  # How strongly we expect reversion
            reversion_prediction = current_price - deviation * reversion_strength

        # 5. Order book slope model (new)
        slope_prediction = current_price
        if "ask_slope" in recent[-1] and "bid_slope" in recent[-1]:
            ask_slope = recent[-1]["ask_slope"]
            bid_slope = recent[-1]["bid_slope"]
            # If ask slope is steep and bid slope is shallow, price likely to decrease
            if abs(ask_slope) > abs(bid_slope) * 1.5:
                slope_prediction = current_price - 2  # Price likely to decrease
            elif abs(bid_slope) > abs(ask_slope) * 1.5:
                slope_prediction = current_price + 2  # Price likely to increase

        # Ensemble the predictions with dynamic weights
        # Adjust weights based on recent prediction accuracy if we had enough history
        ensemble_weights = [0.25, 0.25, 0.2, 0.2, 0.1]  # Initial weights

        # Calculate adaptive weights based on recent performance
        if len(self.squid_ink_features) > 10:
            # Try to adaptively adjust weights based on which models have been performing well
            # This is a simplified version - would need more sophisticated tracking in production
            recent_direction = 1 if momentum > 0 else -1

            # If we've seen consistent directional movement, increase momentum weight
            if all(self.price_history["SQUID_INK"][-3:][i] > self.price_history["SQUID_INK"][-3:][i-1]
                   for i in range(1, 3)) and momentum > 0:
                ensemble_weights[1] = 0.35  # Increase momentum weight
                ensemble_weights[3] = 0.15  # Decrease reversion weight
            elif all(self.price_history["SQUID_INK"][-3:][i] < self.price_history["SQUID_INK"][-3:][i-1]
                     for i in range(1, 3)) and momentum < 0:
                ensemble_weights[1] = 0.35  # Increase momentum weight
                ensemble_weights[3] = 0.15  # Decrease reversion weight

            # If price is far from long-term mean, increase reversion weight
            if len(self.price_history["SQUID_INK"]) > 20:
                long_term_mean = sum(
                    self.price_history["SQUID_INK"][-20:]) / 20
                rel_deviation = abs(
                    current_price - long_term_mean) / long_term_mean
                if rel_deviation > 0.02:  # If price is more than 2% away from mean
                    ensemble_weights[3] = 0.3  # Increase reversion weight
                    # Normalize other weights
                    total = sum(ensemble_weights) - ensemble_weights[3]
                    for i in range(len(ensemble_weights)):
                        if i != 3:
                            ensemble_weights[i] = ensemble_weights[i] * \
                                (1 - 0.3) / total

            # If order imbalance is very strong, increase its weight
            if "volume_imbalance" in recent[-1] and abs(recent[-1]["volume_imbalance"]) > 0.5:
                ensemble_weights[2] = 0.3  # Increase imbalance model weight
                # Normalize other weights
                total = sum(ensemble_weights) - ensemble_weights[2]
                for i in range(len(ensemble_weights)):
                    if i != 2:
                        ensemble_weights[i] = ensemble_weights[i] * \
                            (1 - 0.3) / total

        predicted_price = (
            ma_prediction * ensemble_weights[0] +
            momentum_prediction * ensemble_weights[1] +
            imbalance_prediction * ensemble_weights[2] +
            reversion_prediction * ensemble_weights[3] +
            slope_prediction * ensemble_weights[4]
        )

        # Calculate predicted difference for trading signals
        predicted_diff = predicted_price - current_price

        return predicted_price, predicted_diff
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
