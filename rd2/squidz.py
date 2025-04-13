import json
from typing import Any, Dict, List
import math
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import collections
import statistics
import numpy as np
import pandas as pd



# === Enhancement Utilities (No sklearn) ===
import pandas as pd

def calculate_zscore(prices, window=20):
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return (prices - mean) / std

def calculate_volatility(prices, window=30):
    return prices.diff().abs().rolling(window).mean()

def calculate_trend(prices, short=10, long=50):
    short_ma = prices.rolling(short).mean()
    long_ma = prices.rolling(long).mean()
    return short_ma > long_ma

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

class SMCTrader:
    def __init__(self, num_particles=150, smoothing_factor=0.15, max_drawdown=0.1):
        self.num_particles = num_particles
        # Initialize particles around 1875 (SQUID_INK price seen from chart)
        self.particles = np.random.normal(loc=1875, scale=25, size=num_particles)
        self.weights = np.ones(num_particles) / num_particles  # Equal weights initially
        self.price_history = []  # Track historical prices for resampling
        self.smoothing_factor = smoothing_factor  # Factor to smooth price predictions
        self.max_drawdown = max_drawdown  # Max drawdown limit
        self.last_prediction = 1875  # Default starting prediction
        
        # Market regime tracking
        self.volatility_estimate = 10  # Starting volatility estimate
        self.regime = "normal"  # Market regime: normal, volatile, trending
        self.trend_direction = 0  # 0=neutral, 1=up, -1=down
        
        # Performance tracking
        self.prediction_errors = []  # Track prediction errors

    def update_particles(self, market_data):
        """
        Update particle weights based on new market data.
        """
        # Update price history - handle case with no current_price
        if 'current_price' not in market_data:
            return
            
        current_price = market_data['current_price']
        if current_price is None:
            return
            
        self.price_history.append(current_price)
        if len(self.price_history) > 50:
            self.price_history = self.price_history[-50:]
        
        # Detect market regime
        self._detect_regime()
        
        # Calculate prediction error from last prediction
        if hasattr(self, 'last_prediction'):
            prediction_error = abs(self.last_prediction - current_price)
            self.prediction_errors.append(prediction_error)
            if len(self.prediction_errors) > 20:
                self.prediction_errors = self.prediction_errors[-20:]
        
        # Update particle weights based on how well they predicted the current price
        new_weights = []
        for particle in self.particles:
            # Calculate likelihood
            error = abs(particle - current_price)
            # Use Gaussian likelihood
            # Avoid division by zero
            volatility = max(self.volatility_estimate, 0.1)
            likelihood = np.exp(-0.5 * (error/volatility)**2)
            new_weights.append(likelihood)
        
        # Normalize weights
        total_weight = sum(new_weights)
        if total_weight > 0:
            self.weights = np.array(new_weights) / total_weight
        else:
            # If all weights are zero, reset to uniform
            self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Resample particles based on weights
        self._resample_particles()
        
        # Propagate particles (add noise and drift based on regime)
        self._propagate_particles()

    def _detect_regime(self):
        """
        Detect current market regime based on price history.
        """
        if len(self.price_history) < 10:
            return
        
        # Calculate volatility as rolling std dev
        recent_prices = np.array(self.price_history[-10:])
        self.volatility_estimate = np.std(recent_prices)
        
        # Calculate trend as simple linear regression slope
        x = np.arange(len(recent_prices))
        if len(x) > 1:  # Need at least 2 points
            slope, _ = np.polyfit(x, recent_prices, 1)
            
            # Determine trend direction
            if slope > 0.5:
                self.trend_direction = 1  # Up trend
            elif slope < -0.5:
                self.trend_direction = -1  # Down trend
            else:
                self.trend_direction = 0  # Neutral
        
        # Set regime based on volatility
        if self.volatility_estimate > 15:
            self.regime = "volatile"
        elif abs(self.trend_direction) == 1:
            self.regime = "trending"
        else:
            self.regime = "normal"

    def _resample_particles(self):
        """
        Resample particles based on their weights.
        """
        # Systematic resampling
        cumulative_sum = np.cumsum(self.weights)
        
        # Generate uniform samples
        step = 1.0 / self.num_particles
        u = np.random.uniform(0, step)
        indexes = []
        
        i = 0
        for j in range(self.num_particles):
            while u > cumulative_sum[i]:
                i += 1
                if i >= len(cumulative_sum):
                    i = len(cumulative_sum) - 1
                    break
            indexes.append(i)
            u += step
        
        # Update particles
        self.particles = self.particles[indexes]
        # Reset weights to uniform
        self.weights = np.ones(self.num_particles) / self.num_particles

    def _propagate_particles(self):
        """
        Move particles forward with noise based on current regime.
        """
        current_price = self.price_history[-1]
        
        # Base noise level depends on regime
        if self.regime == "volatile":
            noise_scale = self.volatility_estimate * 1.2
            # In volatile markets, spread particles wider
            attraction_strength = 0.2
        elif self.regime == "trending":
            noise_scale = self.volatility_estimate * 0.8
            # In trending markets, bias particles in trend direction
            attraction_strength = 0.4
        else:  # Normal regime
            noise_scale = self.volatility_estimate * 0.6
            attraction_strength = 0.5
        
        # Add trend component
        trend_component = 0
        if self.regime == "trending":
            # Add stronger bias in trend direction
            trend_component = self.trend_direction * self.volatility_estimate * 0.8
        
        # Mean reversion component (pull towards recent mean)
        if len(self.price_history) >= 20:
            mean_price = np.mean(self.price_history[-20:])
            mean_reversion = (mean_price - self.particles) * 0.1
        else:
            mean_reversion = 0
        
        # Update particles
        noise = np.random.normal(0, noise_scale, self.num_particles)
        attraction = (current_price - self.particles) * attraction_strength
        
        self.particles = self.particles + attraction + noise + trend_component + mean_reversion

    def get_final_prediction(self):
        """
        Get the weighted average of particles as the final prediction, with smoothing.
        """
        if len(self.particles) == 0:
            return self.last_prediction
        
        # Weighted average prediction
        raw_prediction = np.sum(self.particles * self.weights)
        
        # Apply smoothing
        smoothed_prediction = (self.smoothing_factor * raw_prediction + 
                              (1 - self.smoothing_factor) * self.last_prediction)
        
        # Apply additional adjustments based on regime
        if self.regime == "trending":
            # Add trend bias
            trend_bias = self.trend_direction * self.volatility_estimate * 0.5
            smoothed_prediction += trend_bias
        
        # Store for next iteration
        self.last_prediction = smoothed_prediction
        
        return smoothed_prediction

    def predict_price(self, particle, market_data):
        """
        Predict the next price based on particle state and market data.
        """
        # Simple linear adjustment based on regime
        if self.regime == "trending":
            return particle + (self.trend_direction * self.volatility_estimate * 0.2)
        elif self.regime == "volatile":
            # In volatile markets, expect mean reversion
            current_price = market_data['current_price']
            if len(self.price_history) >= 10:
                mean_price = np.mean(self.price_history[-10:])
                # Pull prediction toward mean
                return particle + (mean_price - current_price) * 0.15
            return particle
        else:
            # Normal regime - slight random walk
            return particle + np.random.normal(0, self.volatility_estimate * 0.1)
    

class Trader:
    def __init__(self):
        self.position_limit = 50
        self.target_inventory = 0  # Target inventory level (default: neutral)

        # State variables
        self.smc_trader = SMCTrader(num_particles=100, smoothing_factor=0.2, max_drawdown=0.1)
        self.diffeq_memory = {}
        self.entry_prices = {}  # Dictionary to track entry prices for each symbol
        self.trailing_stops = {}
        self.zscore_history = []
        self.price_history = {}  # Initialize as empty dict
        self.volatility_estimates = {}
        self.last_mid_prices = {}
        self.timestamps = {}
        self.fair_value_estimates = {}
        self.realized_pnl = {}
        self.trades_completed = {}

        # KELP specific parameters
        self.last_mid_prices = {}  # To store last known mid prices

        # SQUID_INK specific parameters
        self.squid_ink_features = []
        self.squid_ink_trades = []  # Recent trade data for SQUID_INK
        self.squid_ink_windows = [3, 5, 10]
        self.squid_ink_lags = [1, 2, 3]

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Enhanced trading strategy with pattern recognition for SQUID_INK
        """
        try:
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


            # Process SQUID_INK with enhanced pattern recognition strategy
            if "SQUID_INK" in state.order_depths:
                self.process_squid_ink_enhanced(state, result)

            # No conversions in tutorial round
            
            # Track recent market trades for SQUID_INK
            if "SQUID_INK" in state.market_trades:
                self.squid_ink_trades.extend(state.market_trades["SQUID_INK"])
                if len(self.squid_ink_trades) > 100:
                    self.squid_ink_trades = self.squid_ink_trades[-100:]
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
                "squid_ink_features": self.squid_ink_features[:5],  # Only store a few features to avoid overflow
            })

            # Return results
            logger.flush(state, result, conversions, trader_data)
            return result, conversions, trader_data
        except Exception as e:
            # If any error occurs, log it and return empty result
            logger.print(f"ERROR in run: {str(e)}")
            return {}, 0, ""


    def process_squid_ink_enhanced(self, state: TradingState, result: dict) -> None:
        """
        Enhanced SQUID_INK trading strategy with improved SMC integration and risk management.
        """
        symbol = "SQUID_INK"
        position_limit = 40  # Position limit for SQUID_INK
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
            spread = best_sell_price - best_buy_price
        elif best_sell_price:
            mid_price = best_sell_price
            spread = 2  # Assume default spread
        elif best_buy_price:
            mid_price = best_buy_price
            spread = 2  # Assume default spread
        else:
            mid_price = self.last_mid_prices.get(symbol, 1875)  # Default based on chart data
            spread = 2

        # Update price history for SQUID_INK
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(mid_price)
        # Limit history size to 100 data points
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]

        # OPTIONAL: Update optimal thresholds if we have enough history (e.g., 50 data points)
        if len(self.price_history[symbol]) >= 50:
            self.set_optimal_thresholds(pd.Series(self.price_history[symbol]))

        # STEP 1: UPDATE SMC MODEL
        market_data = {
            'current_price': mid_price,
            'spread': spread,
            'best_bid': best_buy_price,
            'best_ask': best_sell_price,
            'position': current_position
        }
        self.smc_trader.update_particles(market_data)
        smc_prediction = self.smc_trader.get_final_prediction()

        # STEP 2: CALCULATE ADVANCED INDICATORS
        volatility = np.std(self.price_history[symbol][-20:]) if len(self.price_history[symbol]) >= 20 else 10
        long_term_mean = np.mean(self.price_history[symbol][-50:]) if len(self.price_history[symbol]) >= 50 else mid_price
        z_score = (mid_price - long_term_mean) / volatility if volatility > 0 else 0
        short_term_change = 0
        if len(self.price_history[symbol]) >= 5:
            short_term_change = self.price_history[symbol][-1] - self.price_history[symbol][-5]
        spike_detected = False
        if len(self.price_history[symbol]) >= 3:
            last_moves = [self.price_history[symbol][i] - self.price_history[symbol][i-1] 
                        for i in range(-1, -3, -1)]
            spike_detected = (last_moves[0] * last_moves[1] < 0) and (abs(last_moves[0]) > volatility)

        self.update_squid_ink_features(state, symbol, mid_price)
        ensemble_price, price_diff = self.predict_with_ensemble(mid_price)

        # STEP 3: COMBINE PREDICTIONS WITH WEIGHTS
        ensemble_weight = max(0.2, min(0.7, 1.0 - volatility/30))
        smc_weight = 1.0 - ensemble_weight
        combined_prediction = ensemble_price * ensemble_weight + smc_prediction * smc_weight

        # STEP 4: DETERMINE ADAPTIVE THRESHOLD BASED ON MARKET CONDITIONS
        position_factor = abs(current_position) / position_limit  # How full our position is
        vol_factor = volatility / 15  # Normalized volatility
        if hasattr(self, 'optimal_threshold'):
            adjusted_threshold = self.optimal_threshold * (1 + 0.5 * position_factor + 0.5 * vol_factor)
        else:
            base_threshold = max(1.5, min(5, volatility * 0.75))
            adjusted_threshold = base_threshold * (1 + 0.5 * position_factor + 0.5 * vol_factor)

        # Calculate acceptable prices
        fair_value = combined_prediction
        acc_bid = int(fair_value - adjusted_threshold)
        acc_ask = int(fair_value + adjusted_threshold)

        # STEP 5: IMPROVED RISK MANAGEMENT
        if abs(current_position) > position_limit * 0.7:
            if current_position > 0:
                acc_ask -= 2  # Lower sell price to unwind long position
            else:
                acc_bid += 2  # Raise buy price to unwind short position

        # STEP 6: EXECUTION LOGIC WITH IMPROVED SIZING
        position = current_position
        for price, volume in sell_orders.items():
            if (price < acc_bid or (current_position < -10 and price <= fair_value)) and position < position_limit:
                price_discount = (acc_bid - price) / volatility
                size_factor = min(1.0, 0.5 + (price_discount * 0.5))
                if z_score < -1.5:
                    size_factor = min(1.0, size_factor * 1.3)
                order_size = min(-volume, int((position_limit - position) * size_factor))
                if order_size > 0:
                    squid_orders.append(Order(symbol, price, order_size))
                    position += order_size
                    logger.print(f"SQUID OPPORTUNITY BUY: {order_size} @ {price} (z={z_score:.2f}, vol={volatility:.2f})")

        undercut_buy = best_buy_price + 1 if best_buy_price else acc_bid - 1
        undercut_sell = best_sell_price - 1 if best_sell_price else acc_ask + 1

        bid_price = min(undercut_buy, acc_bid)
        if z_score > 1.0 and volatility > 12:
            bid_price = min(bid_price, acc_bid - 1)
        if z_score < -1.5:
            bid_size = min(30, position_limit - position)
        elif spike_detected and short_term_change < 0:
            bid_size = min(25, position_limit - position)
        elif ensemble_price > mid_price + volatility:
            bid_size = min(35, position_limit - position)
        else:
            confidence = min(1.0, max(0.4, 1.0 - abs(z_score) * 0.15))
            bid_size = min(15, int((position_limit - position) * confidence))
        if bid_size > 0 and position < position_limit and current_position < position_limit * 0.8:
            squid_orders.append(Order(symbol, bid_price, bid_size))
            position += bid_size

        position = current_position
        for price, volume in buy_orders.items():
            if (price > acc_ask or (current_position > 10 and price >= fair_value)) and position > -position_limit:
                price_premium = (price - acc_ask) / volatility
                size_factor = min(1.0, 0.5 + (price_premium * 0.5))
                if z_score > 1.5:
                    size_factor = min(1.0, size_factor * 1.3)
                order_size = max(-volume, int((-position_limit - position) * size_factor))
                if order_size < 0:
                    squid_orders.append(Order(symbol, price, order_size))
                    position += order_size
                    logger.print(f"SQUID OPPORTUNITY SELL: {-order_size} @ {price} (z={z_score:.2f}, vol={volatility:.2f})")

        ask_price = max(undercut_sell, acc_ask)
        if z_score < -1.0 and volatility > 12:
            ask_price = max(ask_price, acc_ask + 1)
        if z_score > 1.5:
            ask_size = max(-30, -position_limit - position)
        elif spike_detected and short_term_change > 0:
            ask_size = max(-25, -position_limit - position)
        elif ensemble_price < mid_price - volatility:
            ask_size = max(-35, -position_limit - position)
        else:
            confidence = min(1.0, max(0.2, 1.0 - abs(z_score) * 0.2))
            ask_size = max(-15, int((-position_limit - position) * confidence))
        if ask_size < 0 and position > -position_limit and current_position > -position_limit * 0.8:
            squid_orders.append(Order(symbol, ask_price, ask_size))
            position += ask_size

        # STEP 7: ACTIVE POSITION MANAGEMENT
        if abs(z_score) > 2.0 and volatility < 15:
            target_position = int(-z_score * 10)
            target_position = max(-position_limit * 0.8, min(position_limit * 0.8, target_position))
            position_delta = target_position - current_position
            if abs(position_delta) > 5:
                if position_delta > 0 and position < position_limit:
                    extra_buy_size = min(position_delta, position_limit - position)
                    extra_buy_price = max(acc_bid - 1, min(undercut_buy, acc_bid))
                    if extra_buy_size > 0:
                        squid_orders.append(Order(symbol, extra_buy_price, extra_buy_size))
                        logger.print(f"SQUID POSITION ADJUSTMENT BUY: {extra_buy_size} @ {extra_buy_price}")
                elif position_delta < 0 and position > -position_limit:
                    extra_sell_size = max(position_delta, -position_limit - position)
                    extra_sell_price = min(acc_ask + 1, max(undercut_sell, acc_ask))
                    if extra_sell_size < 0:
                        squid_orders.append(Order(symbol, extra_sell_price, extra_sell_size))
                        logger.print(f"SQUID POSITION ADJUSTMENT SELL: {-extra_sell_size} @ {extra_sell_price}")

        # STEP 8: TRAILING STOP LOGIC FOR RISK MANAGEMENT
        if current_position > 10:
            if len(self.price_history[symbol]) >= 10:
                recent_high = max(self.price_history[symbol][-10:])
                stop_price = int(recent_high * 0.99)
                if mid_price < stop_price and mid_price < long_term_mean:
                    stop_size = min(current_position // 2, position_limit)
                    if stop_size > 0:
                        squid_orders.append(Order(symbol, best_buy_price, -stop_size))
                        logger.print(f"SQUID TRAILING STOP SELL: {stop_size} @ {best_buy_price}")
        elif current_position < -10:
            if len(self.price_history[symbol]) >= 10:
                recent_low = min(self.price_history[symbol][-10:])
                stop_price = int(recent_low * 1.01)
                if mid_price > stop_price and mid_price > long_term_mean:
                    stop_size = min(abs(current_position) // 2, position_limit)
                    if stop_size > 0:
                        squid_orders.append(Order(symbol, best_sell_price, stop_size))
                        logger.print(f"SQUID TRAILING STOP BUY: {stop_size} @ {best_sell_price}")

        logger.print(f"===== {symbol} Enhanced Trading =====")
        logger.print(f"Price: {mid_price:.2f}, SMC Pred: {smc_prediction:.2f}, Ensemble: {ensemble_price:.2f}")
        logger.print(f"Combined: {combined_prediction:.2f}, Fair Value: {fair_value:.2f}, Z-Score: {z_score:.2f}")
        logger.print(f"Volatility: {volatility:.2f}, Momentum: {short_term_change:.2f}, Spike: {spike_detected}")
        logger.print(f"Threshold: {adjusted_threshold:.2f}, Bid: {acc_bid}, Ask: {acc_ask}")
        logger.print(f"Position: {current_position}/{position_limit}")
        logger.print(f"Orders: {len(squid_orders)}")

        self.last_mid_prices[symbol] = mid_price
        if not hasattr(self, 'volatility_estimates'):
            self.volatility_estimates = {}
        self.volatility_estimates[symbol] = volatility
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
        ma_weights = [0.5, 0.3, 0.2]  # Most recent has highest weight
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
        ensemble_weights = [0.35, 0.25, 0.2, 0.1, 0.1]  # Initial weights

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
    
    def grid_search_optimal_thresholds(self, price_series: pd.Series, 
                                    entry_range=(-3.0, -1.0), exit_range=(1.0, 3.0), 
                                    steps=5) -> Dict[str, float]:
        """
        Grid search for optimal entry and exit z-score thresholds using historical price data,
        without running a full trading simulation.

        Parameters:
            price_series: pd.Series
                The historical price data to use for evaluation.
            entry_range: tuple(float, float)
                The range of entry thresholds (for longs) to try (e.g., between -3 and -1).
            exit_range: tuple(float, float)
                The range of exit thresholds (for shorts) to try (e.g., between 1 and 3).
            steps: int
                Number of grid steps in each range.

        Returns:
            A dictionary with the best thresholds:
                {'entry_threshold': optimal_entry, 'exit_threshold': optimal_exit}
        """
        best_score = -np.inf
        best_params = {'entry_threshold': entry_range[0], 'exit_threshold': exit_range[0]}

        entry_values = np.linspace(entry_range[0], entry_range[1], steps)
        exit_values = np.linspace(exit_range[0], exit_range[1], steps)
        
        # Calculate the rolling z-score once to use in evaluation
        z_series = calculate_zscore(price_series)

        # Loop through grid of candidate threshold pairs
        for entry_thresh in entry_values:
            for exit_thresh in exit_values:
                # Identify potential buy (long) and sell (short) signals based on thresholds.
                buy_signals = z_series[z_series < entry_thresh]
                sell_signals = z_series[z_series > exit_thresh]

                # Simplified objective:
                # For buy signals (which are negative) we take the absolute mean (i.e. more negative is better),
                # and for sell signals (positive) we take the mean.
                if not buy_signals.empty and not sell_signals.empty:
                    score_buy = -buy_signals.mean()
                    score_sell = sell_signals.mean()
                    score = score_buy + score_sell
                else:
                    score = 0

                if score > best_score:
                    best_score = score
                    best_params['entry_threshold'] = entry_thresh
                    best_params['exit_threshold'] = exit_thresh

        return best_params


    def set_optimal_thresholds(self, price_series: pd.Series):
        """
        Perform grid search and store optimal z-score thresholds as instance variables.
        """
        optimal = self.grid_search_optimal_thresholds(price_series)
        self.optimal_entry_threshold = optimal['entry_threshold']
        self.optimal_exit_threshold = optimal['exit_threshold']
        # Combine these into a single threshold value if desired (here as an average)
        self.optimal_threshold = (abs(self.optimal_entry_threshold) + abs(self.optimal_exit_threshold)) / 2
        logger.print(f"Optimal thresholds set: Entry {self.optimal_entry_threshold}, Exit {self.optimal_exit_threshold}")

