import json
from typing import Any, Dict, List
import math
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import collections
import numpy as np
import pandas as pd

# Simple rolling z-score calculation
def rolling_zscore(prices: pd.Series, window: int) -> pd.Series:
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return (prices - mean) / (std.replace(0, 1))

# Logger for printing & flushing
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750
    def print(self, *objs: Any, sep: str=" ", end: str="\n") -> None:
        self.logs += sep.join(map(str, objs)) + end
    def flush(self, state: TradingState, orders: dict[Symbol, List[Order]], conversions: int, data: str) -> None:
        out = json.dumps([state.timestamp, data, orders, conversions, self.logs], cls=ProsperityEncoder, separators=(",", ":"))
        print(out)
        self.logs = ""

logger = Logger()

# A minimal SMC-based predictive module
class SMCTrader:
    def __init__(self, n_particles=100):
        self.n = n_particles
        self.particles = np.random.normal(1875, 25, n_particles)
        self.weights = np.ones(n_particles) / n_particles
        self.price_history = []
        self.last_pred = 1875
        self.vol_est = 10
    def update(self, price: float):
        if price is None: 
            return
        self.price_history.append(price)
        self.price_history = self.price_history[-50:]
        if len(self.price_history) >= 10:
            self.vol_est = np.std(self.price_history[-10:])
        diffs = np.abs(self.particles - price)
        vol = max(self.vol_est, 0.1)
        w = np.exp(-0.5*(diffs/vol)**2)
        w /= w.sum() if w.sum()>0 else 1
        self.weights = w
        idx = np.random.choice(range(self.n), size=self.n, p=w)
        self.particles = self.particles[idx]
        # minimal propagation
        noise = np.random.normal(0, vol*0.5, self.n)
        drift = (price - self.particles)*0.3
        self.particles += drift + noise
    def predict(self):
        raw = (self.particles * self.weights).sum()
        pred = 0.2*raw + 0.8*self.last_pred
        self.last_pred = pred
        return pred

# Main Trader implementing "squid sauce":
# - rolling z-score in windows 10-20 to find vol clusters
# - directional trading based on grid-searched z-score thresholds
class Trader:
    def __init__(self):
        self.smc = SMCTrader(n_particles=100)
        self.price_data: Dict[str, List[float]] = {}
        self.optimal_window = 20
        self.optimal_threshold = 2.0
        self.position_limit = 50
    def grid_search_window(self, series: pd.Series, candidates=[10,15,20]) -> int:
        best_w, best_metric = candidates[0], None
        for w in candidates:
            z = rolling_zscore(series, w)
            # simple metric: lower variance of z => stable
            m = z.var()
            if best_metric is None or m < best_metric:
                best_metric = m
                best_w = w
        return best_w
    def grid_search_thresholds(self, series: pd.Series) -> float:
        # We'll just pick a single threshold => symmetrical for buy/sell
        best_val, best_score = 1.5, -np.inf
        for thr in np.linspace(1.0, 3.0, 5):
            z = rolling_zscore(series, self.optimal_window)
            buy_signals = z[z < -thr]
            sell_signals = z[z > thr]
            score = (-buy_signals.mean() if not buy_signals.empty else 0) + (sell_signals.mean() if not sell_signals.empty else 0)
            if score>best_score:
                best_score = score
                best_val = thr
        return best_val
    def process_squid_ink(self, state: TradingState, res: dict):
        sym = "SQUID_INK"
        od = state.order_depths[sym]
        cur_pos = state.position.get(sym, 0)
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        if best_ask and best_bid:
            mid = 0.5*(best_ask+best_bid)
        elif best_ask:
            mid = best_ask
        elif best_bid:
            mid = best_bid
        else:
            mid = self.price_data.get(sym, [1875])[-1]
        self.price_data.setdefault(sym,[]).append(mid)
        self.price_data[sym] = self.price_data[sym][-200:]
        if len(self.price_data[sym])>50:
            s = pd.Series(self.price_data[sym])
            self.optimal_window = self.grid_search_window(s, [10,15,20])
            self.optimal_threshold = self.grid_search_thresholds(s)
        # Update SMC
        self.smc.update(mid)
        pred = self.smc.predict()
        # Use the chosen window for rolling z-score
        window_z = rolling_zscore(pd.Series(self.price_data[sym]), self.optimal_window)
        z_val = window_z.iloc[-1] if not window_z.empty else 0
        # trade if z < -threshold => buy, z > threshold => sell
        orders = []
        limit = 20
        if z_val < -self.optimal_threshold and cur_pos<self.position_limit:
            buy_size = min(self.position_limit-cur_pos, limit)
            orders.append(Order(sym, int(pred-2), buy_size))
        elif z_val> self.optimal_threshold and cur_pos>-self.position_limit:
            sell_size = min(cur_pos+self.position_limit, limit)
            orders.append(Order(sym, int(pred+2), -sell_size))
        res[sym] = orders
    def run(self, state: TradingState) -> tuple[dict[Symbol, List[Order]], int, str]:
        try:
            result = {}
            if "SQUID_INK" in state.order_depths:
                self.process_squid_ink(state, result)
            data = json.dumps({"zscore_window": self.optimal_window, "threshold":self.optimal_threshold})
            logger.flush(state, result, 0, data)
            return result, 0, data
        except Exception as e:
            logger.print("ERROR:",str(e))
            return {},0,""