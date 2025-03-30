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
        # Configuration parameters - using your successful parameters
        self.position_limit = 50
        
        # Parameters for trading strategy
        self.params = {
            "RAINFOREST_RESIN": {
                "take_width": 0,  # Zero threshold for taking mispriced orders
                "clear_width": 1,  # Width for position clearing
                "disregard_edge": 1,  # Ignore orders this close to fair value
                "join_edge": 2,  # Join orders within this edge
                "default_edge": 3,  # Default edge when no orders to join/penny
                "soft_position_limit": 40,  # Trigger position management
            },
            "KELP": {
                "take_width": 1,
                "clear_width": 1,
                "disregard_edge": 1,
                "join_edge": 2,
                "default_edge": 3,
                "soft_position_limit": 40,
            }
        }
        
        # State variables for tracking market data
        self.price_history = {}
        self.last_mid_prices = {}
        self.fair_values = {}
    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Hybrid strategy combining aggressive order taking with intelligent market making
        """
        # Initialize result dictionary
        result = {}
        
        # Initialize or load stored data
        stored_data = {}
        if state.traderData:
            try:
                stored_data = json.loads(state.traderData)
                self.price_history = stored_data.get("price_history", {})
                self.last_mid_prices = stored_data.get("last_mid_prices", {})
                self.fair_values = stored_data.get("fair_values", {})
            except:
                pass
        
        # Process RAINFOREST_RESIN
        if "RAINFOREST_RESIN" in state.order_depths:
            symbol = "RAINFOREST_RESIN"
            order_depth = state.order_depths[symbol]
            current_position = state.position.get(symbol, 0)
            
            # Calculate fair value (simple mid-price model for now)
            fair_value = self.calculate_fair_value(symbol, order_depth)
            self.fair_values[symbol] = fair_value
            
            # Track price history for analysis
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(fair_value)
            if len(self.price_history[symbol]) > 50:
                self.price_history[symbol] = self.price_history[symbol][-50:]
            
            # Initialize tracking for order volumes
            buy_order_volume = 0
            sell_order_volume = 0
            
            # Step 1: Take mispriced orders
            take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                symbol, 
                order_depth,
                fair_value,
                self.params[symbol]["take_width"],
                current_position,
                buy_order_volume,
                sell_order_volume
            )
            
            # Step 2: Clear excessive positions
            clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                symbol,
                order_depth,
                fair_value,
                self.params[symbol]["clear_width"],
                current_position,
                buy_order_volume,
                sell_order_volume
            )
            
            # Step 3: Make markets with strategic placement
            make_orders, buy_order_volume, sell_order_volume = self.make_orders(
                symbol,
                order_depth,
                fair_value,
                current_position,
                buy_order_volume,
                sell_order_volume,
                self.params[symbol]["disregard_edge"],
                self.params[symbol]["join_edge"],
                self.params[symbol]["default_edge"],
                True,  # Enable position management
                self.params[symbol]["soft_position_limit"]
            )
            
            # Combine all orders
            all_orders = take_orders + clear_orders + make_orders
            
            # Log strategy decisions
            logger.print(f"===== {symbol} Trading Strategy =====")
            logger.print(f"Fair Value: {fair_value}, Position: {current_position}")
            logger.print(f"Take Orders: {len(take_orders)}, Clear Orders: {len(clear_orders)}, Make Orders: {len(make_orders)}")
            
            # Store orders in result
            result[symbol] = all_orders
            
            # Store mid price for next iteration
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                self.last_mid_prices[symbol] = mid_price
        
        # Initialize empty order list for KELP (not trading in tutorial)
        if "KELP" in state.order_depths:
            result["KELP"] = []
        
        # No conversions in tutorial round
        conversions = 0
        
        # Store data for next round
        stored_data = {
            "price_history": self.price_history,
            "last_mid_prices": self.last_mid_prices,
            "fair_values": self.fair_values
        }
        trader_data = json.dumps(stored_data)
        
        # Return results
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    
    def calculate_fair_value(self, symbol: str, order_depth: OrderDepth) -> float:
        """Calculate fair value for a symbol"""
        # Simple mid price calculation
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        elif order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        elif order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        else:
            # If no orders, use last known fair value or default
            return self.fair_values.get(symbol, 10000)
    
    def take_orders(self, symbol: str, order_depth: OrderDepth, fair_value: float, 
                   take_width: float, position: int, buy_order_volume: int, 
                   sell_order_volume: int) -> tuple[List[Order], int, int]:
        """Take mispriced orders from the book"""
        orders = []
        
        # Check sell side (buying opportunities)
        if order_depth.sell_orders:
            # Sort by price (lowest first for buying)
            for price, volume in sorted(order_depth.sell_orders.items()):
                # Skip if price is above fair_value + take_width
                if price > fair_value + take_width:
                    continue
                
                # Calculate how much we can buy
                available_to_buy = min(
                    abs(volume),  # Available volume (negative for sells)
                    self.position_limit - position - buy_order_volume  # Remaining capacity
                )
                
                # Place buy order if we have capacity
                if available_to_buy > 0:
                    orders.append(Order(symbol, price, available_to_buy))
                    buy_order_volume += available_to_buy
                    
                    # Log the trade
                    logger.print(f"TAKE BUY: {available_to_buy} @ {price}, Fair: {fair_value}")
        
        # Check buy side (selling opportunities)
        if order_depth.buy_orders:
            # Sort by price (highest first for selling)
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                # Skip if price is below fair_value - take_width
                if price < fair_value - take_width:
                    continue
                
                # Calculate how much we can sell
                available_to_sell = min(
                    volume,  # Available volume (positive for buys)
                    self.position_limit + position - sell_order_volume  # Remaining capacity
                )
                
                # Place sell order if we have capacity
                if available_to_sell > 0:
                    orders.append(Order(symbol, price, -available_to_sell))
                    sell_order_volume += available_to_sell
                    
                    # Log the trade
                    logger.print(f"TAKE SELL: {available_to_sell} @ {price}, Fair: {fair_value}")
        
        return orders, buy_order_volume, sell_order_volume
    
    def clear_orders(self, symbol: str, order_depth: OrderDepth, fair_value: float,
                    clear_width: float, position: int, buy_order_volume: int,
                    sell_order_volume: int) -> tuple[List[Order], int, int]:
        """Clear excessive positions"""
        orders = []
        
        # Position after take orders
        position_after_take = position + buy_order_volume - sell_order_volume
        
        # Only clear if position is significantly unbalanced
        if abs(position_after_take) < self.params[symbol]["soft_position_limit"]:
            return orders, buy_order_volume, sell_order_volume
        
        # Calculate fair price levels for clearing
        fair_for_bid = round(fair_value - clear_width)
        fair_for_ask = round(fair_value + clear_width)
        
        # Calculate remaining capacity
        buy_capacity = self.position_limit - (position + buy_order_volume)
        sell_capacity = self.position_limit + (position - sell_order_volume)
        
        if position_after_take > 0:
            # We are long, need to sell
            # Check if there are buy orders at or above our fair ask price
            eligible_buys = {price: vol for price, vol in order_depth.buy_orders.items() 
                           if price >= fair_for_ask}
            
            if eligible_buys:
                # Determine how much to clear
                clear_volume = min(
                    sum(eligible_buys.values()),  # Total eligible volume
                    position_after_take,  # Current position
                    sell_capacity  # Remaining sell capacity
                )
                
                if clear_volume > 0:
                    # Place at the fair ask price
                    orders.append(Order(symbol, fair_for_ask, -clear_volume))
                    sell_order_volume += clear_volume
                    
                    # Log the trade
                    logger.print(f"CLEAR SELL: {clear_volume} @ {fair_for_ask}, Position: {position_after_take}")
                    
        elif position_after_take < 0:
            # We are short, need to buy
            # Check if there are sell orders at or below our fair bid price
            eligible_sells = {price: vol for price, vol in order_depth.sell_orders.items() 
                            if price <= fair_for_bid}
            
            if eligible_sells:
                # Determine how much to clear
                clear_volume = min(
                    sum(abs(vol) for vol in eligible_sells.values()),  # Total eligible volume
                    abs(position_after_take),  # Current position
                    buy_capacity  # Remaining buy capacity
                )
                
                if clear_volume > 0:
                    # Place at the fair bid price
                    orders.append(Order(symbol, fair_for_bid, clear_volume))
                    buy_order_volume += clear_volume
                    
                    # Log the trade
                    logger.print(f"CLEAR BUY: {clear_volume} @ {fair_for_bid}, Position: {position_after_take}")
        
        return orders, buy_order_volume, sell_order_volume
    
    def make_orders(self, symbol: str, order_depth: OrderDepth, fair_value: float,
                   position: int, buy_order_volume: int, sell_order_volume: int,
                   disregard_edge: float, join_edge: float, default_edge: float,
                   manage_position: bool = False, soft_position_limit: int = 0
                   ) -> tuple[List[Order], int, int]:
        """Make markets with strategic placement"""
        orders = []
        
        # Find orders to join or penny
        asks_above_fair = [price for price in order_depth.sell_orders.keys() 
                         if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() 
                         if price < fair_value - disregard_edge]
        
        # Determine best prices to join or penny
        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None
        
        # Calculate ask price
        ask = round(fair_value + default_edge)
        if best_ask_above_fair:
            # If there's a nearby ask, join it
            if best_ask_above_fair - fair_value <= join_edge:
                ask = best_ask_above_fair  # Join
            else:
                ask = best_ask_above_fair - 1  # Penny (improve by 1)
        
        # Calculate bid price
        bid = round(fair_value - default_edge)
        if best_bid_below_fair:
            # If there's a nearby bid, join it
            if fair_value - best_bid_below_fair <= join_edge:
                bid = best_bid_below_fair  # Join
            else:
                bid = best_bid_below_fair + 1  # Penny (improve by 1)
        
        # Apply position management if enabled
        if manage_position:
            position_after_take = position + buy_order_volume - sell_order_volume
            
            # If long, make more aggressive asks
            if position_after_take > soft_position_limit:
                ask -= 1
            # If short, make more aggressive bids
            elif position_after_take < -soft_position_limit:
                bid += 1
        
        # Calculate order sizes based on remaining capacity
        buy_size = self.position_limit - (position + buy_order_volume)
        sell_size = self.position_limit + (position - sell_order_volume)
        
        # Place bid if we have capacity
        if buy_size > 0:
            orders.append(Order(symbol, bid, buy_size))
            logger.print(f"MAKE BUY: {buy_size} @ {bid}")
        
        # Place ask if we have capacity
        if sell_size > 0:
            orders.append(Order(symbol, ask, -sell_size))
            logger.print(f"MAKE SELL: {sell_size} @ {ask}")
        
        return orders, buy_order_volume, sell_order_volume