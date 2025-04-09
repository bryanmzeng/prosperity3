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
        self.kelp_cache = []
        self.last_mid_prices = {}  # To store last known mid prices
        
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Enhanced trading strategy with linear regression model for KELP
        """
        result = {}
        timestamp = state.timestamp
        conversions = 0
        
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
            self.process_kelp_conversion_strategy(state, result)
        
        
        
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
        Process KELP using regression model similar to BANANAS strategy
        """
        symbol = "KELP"
        position_limit = 50  # Position limit for KELP

        # Initialize KELP cache if not exists
        if not hasattr(self, 'kelp_cache'):
            self.kelp_cache = []

        # Sort order books for calculating mid price
        order_depth = state.order_depths[symbol]
        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        # Calculate mid price
        best_sell_price = min(sell_orders.keys()) if sell_orders else None
        best_buy_price = max(buy_orders.keys()) if buy_orders else None

        if best_sell_price and best_buy_price:
            mid_price = (best_sell_price + best_buy_price) / 2
        elif best_sell_price:
            mid_price = best_sell_price
        elif best_buy_price:
            mid_price = best_buy_price
        else:
            # No orders on either side, use last known price or default
            mid_price = self.last_mid_prices.get(symbol, 2000)  # Default for KELP

        # Manage KELP cache similar to BANANAS strategy
        if len(self.kelp_cache) == self.kelp_dim:
            self.kelp_cache.pop(0)
        self.kelp_cache.append(mid_price)

        # Predict next price using KELP-specific coefficients
        if len(self.kelp_cache) == self.kelp_dim:
            next_price = self.calc_next_price_kelp()
        else:
            next_price = mid_price

        # Define acceptable price bounds
        acc_bid = int(next_price - 1)
        acc_ask = int(next_price + 1)

        # Compute orders using regression method
        # Note the change in argument order
        kelp_orders = self.compute_orders_regression(
            symbol, state, acc_bid, acc_ask, position_limit
        )

        # Log strategy information
        logger.print(f"===== {symbol} Regression-based Trading =====")
        logger.print(f"Current Mid Price: {mid_price:.2f}, Predicted Next: {next_price:.2f}")
        logger.print(f"Price Bounds - Lower: {acc_bid}, Upper: {acc_ask}")
        logger.print(f"Market - Best Bid: {best_buy_price}, Best Ask: {best_sell_price}")
        logger.print(f"Position: {state.position.get(symbol, 0)}/{position_limit}")
        logger.print(f"Orders: {len(kelp_orders)}")

        # Store current mid price for future reference
        self.last_mid_prices[symbol] = mid_price

        # Add orders to result
        result[symbol] = kelp_orders

    def calc_next_price_kelp(self):
        """
        Predict the next KELP price using a regression model - got coefficients from data analysis on cleaned data
        """
        # KELP-specific coefficients (modify these based on your analysis)
        coef = [0.198536, 0.204951, 0.260282, 0.335230]
        intercept = 2.024153
        
        # Prediction using recent cache values
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
    def process_kelp_conversion_strategy(self, state: TradingState, result: dict) -> tuple[list[Order], int]:
        """
        KELP trading strategy focusing on conversion arbitrage
        """
        symbol = "KELP"
        position_limit = 50

        # Prepare for conversion tracking
        conversions = 0
        
        # Order depth and current market state
        order_depth = state.order_depths[symbol]
        current_position = state.position.get(symbol, 0)

        # Examine conversion observations
        if symbol in state.observations.conversionObservations:
            conversion_obs = state.observations.conversionObservations[symbol]
            
            # Check if conversion is profitable
            # This requires careful examination of:
            # 1. Export/Import tariffs
            # 2. Transport fees
            # 3. Conversion bid/ask prices
            export_cost = conversion_obs.exportTariff
            import_cost = conversion_obs.importTariff
            transport_fee = conversion_obs.transportFees
            
            # Sort order books
            sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
            buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

            # Determine market prices
            best_sell_price = min(sell_orders.keys()) if sell_orders else float('inf')
            best_buy_price = max(buy_orders.keys()) if buy_orders else -float('inf')

            # Conversion profitability check
            # Aim to convert at a price that allows immediate re-shorting profitably
            if best_sell_price != float('inf'):
                # Try to go short at the best sell price
                # Check if conversion back to 0 is profitable
                # This requires careful calculation of all associated costs
                conversion_profit = (conversion_obs.bidPrice - best_sell_price 
                                    - export_cost 
                                    - import_cost 
                                    - transport_fee)
                
                # Aggressive conversion strategy
                # Attempt to go short to the limit if profitable
                if conversion_profit > 0:
                    # Calculate how many units we can convert
                    convert_amount = min(position_limit, abs(current_position))
                    
                    if convert_amount > 0:
                        # Set up short position at best sell price
                        kelp_orders = [
                            Order(symbol, best_sell_price, -convert_amount)
                        ]
                        
                        # Set conversions to move position to 0
                        conversions = convert_amount
                        
                        # Detailed logging
                        logger.print(f"KELP Conversion Arbitrage:")
                        logger.print(f"Shorting {convert_amount} @ {best_sell_price}")
                        logger.print(f"Conversion Profit: {conversion_profit:.4f}")
                        logger.print(f"Export Cost: {export_cost}")
                        logger.print(f"Import Cost: {import_cost}")
                        logger.print(f"Transport Fee: {transport_fee}")
                        
                        # Add orders to result
                        result[symbol] = kelp_orders
                        
                        return kelp_orders, conversions

        # Fallback to standard market making if no conversion opportunity
        # (Similar to previous market-making strategy)
        return self.process_kelp_regression(state, result)
        kelp_orders = []
        
        # Sort order books
        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        # Calculate mid price
        best_sell_price = min(sell_orders.keys()) if sell_orders else None
        best_buy_price = max(buy_orders.keys()) if buy_orders else None

        if best_sell_price and best_buy_price:
            mid_price = (best_sell_price + best_buy_price) / 2
        elif best_sell_price:
            mid_price = best_sell_price
        elif best_buy_price:
            mid_price = best_buy_price
        else:
            mid_price = 7.6128  # Default to long-term mean from OU analysis

        # Conservative market making
        base_order_size = max(5, int(position_limit * 0.2))
        
        # Buy side if under position limit
        if current_position < position_limit:
            buy_size = min(base_order_size, position_limit - current_position)
            bid_price = int(mid_price - 1)  # Slight undercutting
            kelp_orders.append(Order(symbol, bid_price, buy_size))

        # Sell side if over negative position limit
        if current_position > -position_limit:
            sell_size = min(base_order_size, abs(current_position))
            ask_price = int(mid_price + 1)  # Slight overcutting
            kelp_orders.append(Order(symbol, ask_price, -sell_size))

        # Logging
        logger.print(f"KELP Standard Market Making")
        logger.print(f"Mid Price: {mid_price}")
        logger.print(f"Position: {current_position}/{position_limit}")
        logger.print(f"Orders: {len(kelp_orders)}")

        # Add orders to result if any
        if kelp_orders:
            result[symbol] = kelp_orders

        return kelp_orders, conversions