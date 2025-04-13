def run(self, state: TradingState):
    traderObject = {}
    if state.traderData != None and state.traderData != "":
        traderObject = jsonpickle.decode(state.traderData)

    result = {}
    conversions = 0

    # Initialize spread data for PICNIC_BASKET1 from individual components
    if Product.SPREAD not in traderObject:
        traderObject[Product.SPREAD] = {
            "spread_history": [],
            "prev_zscore": 0,
            "clear_flag": False,
            "curr_avg": 0,
        }

    # Initialize spread data for PICNIC_BASKET2 from individual components
    if Product.SPREAD2 not in traderObject:
        traderObject[Product.SPREAD2] = {
            "spread_history": [],
            "prev_zscore": 0,
            "clear_flag": False,
            "curr_avg": 0,
        }
    
    # Initialize spread data for PICNIC_BASKET1 from PICNIC_BASKET2 + components
    if "BASKET1_FROM_BASKET2" not in traderObject:
        traderObject["BASKET1_FROM_BASKET2"] = {
            "basket2_spread_history": [],
            "basket2_prev_zscore": 0,
        }

    basket_position = state.position.get(Product.PICNIC_BASKET1, 0)
    basket2_position = state.position.get(Product.PICNIC_BASKET2, 0)

    # Collect all arbitrage candidates
    arb_candidates = []

    # ETF1 arb from individual components
    spread1_orders = self.spread_orders(
        state.order_depths,
        Product.PICNIC_BASKET1,
        basket_position,
        traderObject[Product.SPREAD],
    )
    if spread1_orders:
        arb_candidates.append(('ETF1', spread1_orders, abs(traderObject[Product.SPREAD]["prev_zscore"])))

    # ETF2 arb from individual components
    spread2_orders = self.spread2_orders(
        state.order_depths,
        Product.PICNIC_BASKET2,
        basket2_position,
        traderObject[Product.SPREAD2],
    )
    if spread2_orders:
        arb_candidates.append(('ETF2', spread2_orders, abs(traderObject[Product.SPREAD2]["prev_zscore"])))
    
    # NEW: ETF1 arb from ETF2 + components
    basket1_from_basket2_orders = self.basket1_from_basket2_orders(
        state.order_depths,
        Product.PICNIC_BASKET1,
        basket_position,
        traderObject["BASKET1_FROM_BASKET2"],
    )
    if basket1_from_basket2_orders:
        arb_candidates.append(
            ('ETF1_FROM_ETF2', 
             basket1_from_basket2_orders, 
             abs(traderObject["BASKET1_FROM_BASKET2"].get("basket2_prev_zscore", 0)))
        )

    # Sort by z-score magnitude (or expected edge)
    arb_candidates.sort(key=lambda x: x[2], reverse=True)

    # Pick top priority arbitrage opportunity
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
        
        elif arb_id == 'ETF1_FROM_ETF2':
            result[Product.CROISSANTS] = arb_orders[Product.CROISSANTS]
            result[Product.JAMS] = arb_orders[Product.JAMS]
            result[Product.DJEMBES] = arb_orders[Product.DJEMBES]
            result[Product.PICNIC_BASKET1] = arb_orders[Product.PICNIC_BASKET1]
            result[Product.PICNIC_BASKET2] = arb_orders[Product.PICNIC_BASKET2]

    # Process RAINFOREST_RESIN if available
    if "RAINFOREST_RESIN" in state.order_depths:
        self.process_rainforest_resin_simple(state, result)
    
    # Process KELP with linear regression model if available
    if "KELP" in state.order_depths:
        self.process_kelp_regression(state, result)

    traderData = jsonpickle.encode(traderObject)

    return result, conversions, traderData
