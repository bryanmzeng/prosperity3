import json
from typing import Any, Dict, List
import math
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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
        
        # Stoikov model parameters
        self.gamma = 0.00000000000005  # Risk aversion parameter (lower = more willing to take risks) - set low rn, seems to perform best like this
        self.time_horizon = 20000  # Approximate trading session duration in ticks
        
        # State variables
        self.price_history = {}
        self.volatility_estimates = {}
        self.last_mid_prices = {}
        self.timestamps = {}
        self.fair_value_estimates = {}
        self.realized_pnl = {}
        self.trades_completed = {}
        
        # Arbitrage configuration
        self.arbitrage_threshold = 0.0000  # 0.08% threshold for arbitrage opportunities
        self.max_arbitrage_size = 50  # Maximum size for arbitrage trades
        
        # Trading configuration
        self.aggressive_mode = True
        self.max_position_per_side = 49  # Maximum position in one direction (avoid hitting limits)
        self.exponential_position_scaling = False  # Scale order sizes exponentially with position
        
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Enhanced Stoikov market making strategy with opportunistic arbitrage
        """
        result = {}
        timestamp = state.timestamp
        
        # Initialize or load stored data
        stored_data = self.initialize_or_load_data(state.traderData)
        
        # Update timestamps
        for symbol in state.order_depths:
            if symbol not in self.timestamps:
                self.timestamps[symbol] = []
            
            self.timestamps[symbol].append(timestamp)
            # Keep only recent timestamps
            if len(self.timestamps[symbol]) > 100:
                self.timestamps[symbol] = self.timestamps[symbol][-100:]
        
        # Process RAINFOREST_RESIN for tutorial round
        if "RAINFOREST_RESIN" in state.order_depths:
            symbol = "RAINFOREST_RESIN"
            order_depth = state.order_depths[symbol]
            current_position = state.position.get(symbol, 0)
            
            # Track this product's trades count
            if symbol not in self.trades_completed:
                self.trades_completed[symbol] = 0
            
            # Get market data
            market_data = self.analyze_market(symbol, order_depth)
            mid_price = market_data["mid_price"]
            
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append(mid_price)
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]
            
            # Update volatility estimate
            self.update_volatility_estimate(symbol)
            
            # Calculate fair value
            fair_value = self.estimate_fair_value(symbol, market_data)
            self.fair_value_estimates[symbol] = fair_value
            
            # 1. FIRST PASS: Look for arbitrage opportunities
            arbitrage_orders = self.capture_arbitrage_opportunities(
                symbol, order_depth, current_position, fair_value
            )
            
            # Calculate remaining capacity after arbitrage
            remaining_buy_capacity = self.max_position_per_side - current_position
            remaining_sell_capacity = self.max_position_per_side + current_position
            
            for order in arbitrage_orders:
                if order.quantity > 0:  # Buy order
                    remaining_buy_capacity -= order.quantity
                else:  # Sell order
                    remaining_sell_capacity -= abs(order.quantity)
            
            # 2. SECOND PASS: Calculate Stoikov quotes with remaining capacity
            stoikov_orders = []
            
            if remaining_buy_capacity > 0 or remaining_sell_capacity > 0:
                # Calculate optimal bid and ask prices using Stoikov model
                bid_price, ask_price, bid_size, ask_size = self.calculate_stoikov_prices(
                    symbol, mid_price, current_position, timestamp
                )
                
                # Adjust sizes based on remaining capacity
                bid_size = min(bid_size, remaining_buy_capacity)
                ask_size = min(ask_size, remaining_sell_capacity)
                
                # Add bid order if we have capacity
                if bid_size > 0:
                    stoikov_orders.append(Order(symbol, bid_price, bid_size))
                
                # Add ask order if we have capacity
                if ask_size > 0:
                    stoikov_orders.append(Order(symbol, ask_price, -ask_size))
            
            # Combine arbitrage and Stoikov orders
            combined_orders = arbitrage_orders + stoikov_orders
            
            # Log strategy information
            self.log_strategy_info(symbol, market_data, current_position, fair_value, combined_orders, arbitrage_orders)
            
            # Store orders in result
            result[symbol] = combined_orders
            
            # Store mid price for next iteration
            self.last_mid_prices[symbol] = mid_price
        
        # Process KELP with joining and pennying strategy
        if "KELP" in state.order_depths:
            symbol = "KELP"
            order_depth = state.order_depths[symbol]
            current_position = state.position.get(symbol, 0)
            
            # Track this product's trades count
            if symbol not in self.trades_completed:
                self.trades_completed[symbol] = 0
            
            # Get market data
            market_data = self.analyze_market(symbol, order_depth)
            
            # Update price history for KELP
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            if "mid_price" in market_data:
                mid_price = market_data["mid_price"]
                self.price_history[symbol].append(mid_price)
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol] = self.price_history[symbol][-100:]
            
            # Initialize orders list
            kelp_orders = []
            
            # Implement joining and pennying strategy for KELP
            if order_depth.buy_orders and order_depth.sell_orders:
                # Get best bid and ask
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                
                # Calculate the spread
                spread = best_ask - best_bid
                
                # Position-aware order sizing
                long_capacity = self.max_position_per_side - current_position
                short_capacity = self.max_position_per_side + current_position
                
                # Base size adjusted by position
                base_size = 10
                position_factor = current_position / self.position_limit
                
                # Adjust size based on position - use less aggressive sizes when nearing limits
                bid_size = max(1, int(base_size * (1 - max(0, position_factor))))
                ask_size = max(1, int(base_size * (1 - max(0, -position_factor))))
                
                # Limit sizes to available capacity
                bid_size = min(bid_size, long_capacity)
                ask_size = min(ask_size, short_capacity)
                
                # Implement pennying strategy when spread is wide enough
                if spread > 2:  # If spread is more than 2 ticks
                    # Penny the best bid if we have buying capacity
                    if bid_size > 0:
                        penny_bid_price = best_bid + 1
                        kelp_orders.append(Order(symbol, penny_bid_price, bid_size))
                    
                    # Penny the best ask if we have selling capacity
                    if ask_size > 0:
                        penny_ask_price = best_ask - 1
                        kelp_orders.append(Order(symbol, penny_ask_price, -ask_size))
                else:
                    # Join the best bid/ask when spread is narrow
                    if bid_size > 0:
                        kelp_orders.append(Order(symbol, best_bid, bid_size))
                    
                    if ask_size > 0:
                        kelp_orders.append(Order(symbol, best_ask, -ask_size))
                
                # Opportunistic deep book orders
                # If there's significant buying pressure, place additional sell orders
                buy_volume = sum([abs(qty) for qty in order_depth.buy_orders.values()])
                sell_volume = sum([abs(qty) for qty in order_depth.sell_orders.values()])
                
                # If book imbalance > 20% and we have capacity, place additional orders
                if buy_volume > sell_volume * 1.2 and short_capacity > ask_size:
                    # Place additional sell orders at higher price
                    extra_ask_size = min(5, short_capacity - ask_size)
                    kelp_orders.append(Order(symbol, best_ask + 2, -extra_ask_size))
                
                # If strong selling pressure, place additional buy orders
                elif sell_volume > buy_volume * 1.2 and long_capacity > bid_size:
                    # Place additional buy orders at lower price
                    extra_bid_size = min(5, long_capacity - bid_size)
                    kelp_orders.append(Order(symbol, best_bid - 2, extra_bid_size))
                
                # Log the KELP strategy info
                logger.print(f"===== {symbol} Joining/Pennying Market Making =====")
                logger.print(f"Best Bid: {best_bid}, Best Ask: {best_ask}, Spread: {spread}")
                logger.print(f"Position: {current_position}/{self.position_limit}")
                logger.print(f"Orders: {len(kelp_orders)}")
                
                # Store mid price for next iteration if available
                if "mid_price" in market_data:
                    self.last_mid_prices[symbol] = market_data["mid_price"]
            
            # Add KELP orders to result
            result[symbol] = kelp_orders
        
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
       
    def initialize_or_load_data(self, trader_data: str) -> Dict:
        """Initialize or load stored data"""
        stored_data = {}
        
        if trader_data:
            try:
                stored_data = json.loads(trader_data)
                
                # Update class variables
                self.price_history = stored_data.get("price_history", {})
                self.volatility_estimates = stored_data.get("volatility_estimates", {})
                self.last_mid_prices = stored_data.get("last_mid_prices", {})
                self.timestamps = stored_data.get("timestamps", {})
                self.fair_value_estimates = stored_data.get("fair_value_estimates", {})
                self.realized_pnl = stored_data.get("realized_pnl", {})
                self.trades_completed = stored_data.get("trades_completed", {})
                
            except Exception as e:
                logger.print(f"Error loading trader data: {e}")
                stored_data = {}
        
        return stored_data
    
    def analyze_market(self, symbol: str, order_depth: OrderDepth) -> Dict:
        """Analyze market data and return key metrics"""
        result = {}
        
        # Calculate best bid and ask
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            result["best_bid"] = best_bid
            result["best_bid_volume"] = order_depth.buy_orders[best_bid]
        else:
            result["best_bid"] = None
            result["best_bid_volume"] = 0
        
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            result["best_ask"] = best_ask
            result["best_ask_volume"] = order_depth.sell_orders[best_ask]
        else:
            result["best_ask"] = None
            result["best_ask_volume"] = 0
        
        # Calculate mid price
        if result["best_bid"] is not None and result["best_ask"] is not None:
            result["mid_price"] = (result["best_bid"] + result["best_ask"]) / 2
            result["spread"] = result["best_ask"] - result["best_bid"]
        elif result["best_bid"] is not None:
            result["mid_price"] = result["best_bid"]
            result["spread"] = 1  # Default when only bid exists
        elif result["best_ask"] is not None:
            result["mid_price"] = result["best_ask"]
            result["spread"] = 1  # Default when only ask exists
        else:
            result["mid_price"] = 10000  # Default if no orders
            result["spread"] = 1
        
        # Calculate order book imbalance
        buy_volume = sum([abs(qty) for qty in order_depth.buy_orders.values()]) if order_depth.buy_orders else 0
        sell_volume = sum([abs(qty) for qty in order_depth.sell_orders.values()]) if order_depth.sell_orders else 0
        
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            result["order_imbalance"] = (buy_volume - sell_volume) / total_volume
        else:
            result["order_imbalance"] = 0
        
        # Store full order book
        result["buy_orders"] = order_depth.buy_orders
        result["sell_orders"] = order_depth.sell_orders
        
        return result
    
    def update_volatility_estimate(self, symbol: str) -> None:
        """Update volatility estimate using recent price history"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            self.volatility_estimates[symbol] = 0.001  # Default low volatility
            return
        
        # Calculate returns
        prices = self.price_history[symbol]
        returns = []
        
        for i in range(1, len(prices)):
            returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        # Calculate volatility as standard deviation of returns
        if returns:
            mean_return = sum(returns) / len(returns)
            squared_deviations = [(r - mean_return) ** 2 for r in returns]
            variance = sum(squared_deviations) / len(returns)
            volatility = math.sqrt(variance)
            
            # Scale volatility to a reasonable range for the model
            scaled_volatility = volatility * prices[-1]
            
            # Update volatility estimate with smoothing
            if symbol in self.volatility_estimates:
                self.volatility_estimates[symbol] = (
                    0.95 * self.volatility_estimates[symbol] + 0.05 * scaled_volatility
                )
            else:
                self.volatility_estimates[symbol] = scaled_volatility
        else:
            self.volatility_estimates[symbol] = 0.001  # Default
    
    def capture_arbitrage_opportunities(self, symbol: str, order_depth: OrderDepth, 
                                     current_position: int, fair_value: float) -> List[Order]:
        """
        Capture arbitrage opportunities by matching mispriced orders in the book
        """
        orders = []
        
        if not self.aggressive_mode:
            return orders
        
        # Check buy side of book (our selling opportunities)
        if order_depth.buy_orders and current_position > -self.max_position_per_side:
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                # Check if price is above our fair value threshold
                if price > fair_value * (1 + self.arbitrage_threshold): #arb threshold at 0, take advantage of any opportunity
                    # Calculate how much we can sell
                    available_to_sell = min(
                        abs(volume),  # Available volume at this price
                        self.max_position_per_side + current_position,  # Position limit
                        self.max_arbitrage_size  # Max trade size
                    )
                    
                    # Only trade if quantity is sufficient
                    if available_to_sell >= 1:
                        orders.append(Order(symbol, price, -available_to_sell))
                        current_position -= available_to_sell  # Update position for next iteration
                        
                        # Log the arbitrage opportunity
                        profit_estimate = (price - fair_value) * available_to_sell
                        logger.print(f"ARBITRAGE SELL: {available_to_sell} @ {price}, " +
                                    f"Fair Value: {fair_value}, Profit: {profit_estimate:.2f}")
                        
                        self.trades_completed[symbol] = self.trades_completed.get(symbol, 0) + 1
        
        # Check sell side of book (our buying opportunities)
        if order_depth.sell_orders and current_position < self.max_position_per_side:
            for price, volume in sorted(order_depth.sell_orders.items()):
                # Check if price is below our fair value threshold
                if price < fair_value * (1 - self.arbitrage_threshold):  #arb threshold at 0, take advantage of any opportunity
                    # Calculate how much we can buy
                    available_to_buy = min(
                        abs(volume),  # Available volume at this price
                        self.max_position_per_side - current_position,  # Position limit
                        self.max_arbitrage_size  # Max trade size
                    )
                    
                    # Only trade if quantity is sufficient
                    if available_to_buy >= 1:
                        orders.append(Order(symbol, price, available_to_buy))
                        current_position += available_to_buy  # Update position for next iteration
                        
                        # Log the arbitrage opportunity
                        profit_estimate = (fair_value - price) * available_to_buy
                        logger.print(f"ARBITRAGE BUY: {available_to_buy} @ {price}, " +
                                    f"Fair Value: {fair_value}, Profit: {profit_estimate:.2f}")
                        
                        self.trades_completed[symbol] = self.trades_completed.get(symbol, 0) + 1
        
        return orders
    
    def estimate_fair_value(self, symbol: str, market_data: Dict) -> float:
        """Estimate fair value using multiple signals"""
        # Start with mid price as base estimate
        mid_price = market_data["mid_price"]
        
        # If we don't have enough data, just use mid price
        if (symbol not in self.price_history or 
            len(self.price_history[symbol]) < 10):
            return mid_price
        
        # Use exponential moving average for more stable estimate
        prices = self.price_history[symbol]
        weights = [0.95 ** i for i in range(min(10, len(prices)))]
        weighted_sum = sum(p * w for p, w in zip(prices[:10], weights))
        ema = weighted_sum / sum(weights)
        
        # Adjust based on order book imbalance
        imbalance = market_data.get("order_imbalance", 0)
        imbalance_adjustment = imbalance * 2  # Scale factor for imbalance impact
        
        # Combine signals (80% EMA, 20% adjusted mid price)
        fair_value = 0.8 * ema + 0.2 * (mid_price + imbalance_adjustment)
        
        return fair_value
    
    def calculate_stoikov_prices(self, symbol: str, mid_price: float, current_position: int, 
                                current_time: int) -> tuple[int, int, int, int]:
        """
        Calculate optimal bid and ask prices using the Stoikov model
        
        Returns: (bid_price, ask_price, bid_size, ask_size)
        """
        # Get volatility estimate
        volatility = self.volatility_estimates.get(symbol, 0.001)
        
        # Calculate inventory risk factor
        q = current_position - self.target_inventory
        
        # Calculate time factor (time remaining in trading session)
        time_remaining = max(1, self.time_horizon - current_time)
        time_factor = time_remaining / self.time_horizon
        
        # Apply Stoikov formula for reservation price
        # r = mid_price - q * gamma * volatility^2 * time_remaining
        reservation_price = mid_price - q * self.gamma * (volatility ** 2) * time_factor
        
        # Calculate optimal spread based on volatility
        optimal_half_spread = self.gamma * (volatility ** 2) * time_factor + (2/self.gamma) * math.log(1 + self.gamma/2)
        
        # Ensure minimum spread
        optimal_half_spread = max(optimal_half_spread, 1.0)
        
        # Calculate raw bid and ask prices
        raw_bid = reservation_price - optimal_half_spread
        raw_ask = reservation_price + optimal_half_spread
        
        # Convert to integer prices
        bid_price = int(raw_bid)
        ask_price = int(raw_ask) + 1  # Ensure ask is above bid
        
        # Position-aware sizing
        position_factor = current_position / self.position_limit
        
        # Base sizes
        base_size = 10
        
        # Scale sizes based on position
        if self.exponential_position_scaling:
            # Exponential scaling provides more aggressive position management
            long_factor = math.exp(min(3, position_factor * 5))
            short_factor = math.exp(min(3, -position_factor * 5))
            
            # Scale order sizes inversely with position direction
            bid_size_scale = 1.0 / long_factor
            ask_size_scale = 1.0 / short_factor
        else:
            # Linear scaling (simpler but less aggressive)
            bid_size_scale = 1.0 - max(0, position_factor * 0.8)
            ask_size_scale = 1.0 - max(0, -position_factor * 0.8)
        
        # Calculate final sizes
        bid_size = max(1, int(base_size * bid_size_scale))
        ask_size = max(1, int(base_size * ask_size_scale))
        
        # Ensure we don't exceed position limits
        bid_size = min(bid_size, self.max_position_per_side - current_position)
        ask_size = min(ask_size, self.max_position_per_side + current_position)
        
        # Ensure sizes are positive
        bid_size = max(0, bid_size)
        ask_size = max(0, ask_size)
        
        return bid_price, ask_price, bid_size, ask_size
    
    def log_strategy_info(self, symbol: str, market_data: Dict, current_position: int, 
                       fair_value: float, orders: List[Order], arbitrage_orders: List[Order]) -> None:
        """Log detailed strategy information"""
        logger.print(f"===== {symbol} Enhanced Stoikov Market Making =====")
        logger.print(f"Mid Price: {market_data['mid_price']:.2f}, Fair Value: {fair_value:.2f}")
        logger.print(f"Position: {current_position}/{self.position_limit}")
        logger.print(f"Volatility Estimate: {self.volatility_estimates.get(symbol, 0):.6f}")
        
        # Log arbitrage activity
        if arbitrage_orders:
            logger.print(f"Arbitrage Orders: {len(arbitrage_orders)}")
            for order in arbitrage_orders:
                direction = "BUY" if order.quantity > 0 else "SELL"
                logger.print(f"  {direction} {abs(order.quantity)} @ {order.price}")
        
        # Log all orders
        logger.print(f"Total Orders: {len(orders)}")
        buy_orders = [o for o in orders if o.quantity > 0]
        sell_orders = [o for o in orders if o.quantity < 0]
        logger.print(f"Buy Orders: {len(buy_orders)}, Sell Orders: {len(sell_orders)}")
        
        # Log order book imbalance
        logger.print(f"Order Imbalance: {market_data.get('order_imbalance', 0):.4f}")
        
        # Log trading statistics
        logger.print(f"Trades Completed: {self.trades_completed.get(symbol, 0)}")
        if symbol in self.realized_pnl:
            logger.print(f"Estimated PnL: {self.realized_pnl[symbol]:.2f}")