import pandas as pd
import datetime as dt

BUY = 1
SELL = -1
NONE = 0

def apply_take_profit(row, PROFIT_FACTOR, SPREAD):
    if row.SIGNAL != NONE:
        if row.SIGNAL == BUY:
            return (row.close - row.open) * PROFIT_FACTOR + row.close + SPREAD
        else:
            return (row.close - row.open) * PROFIT_FACTOR + row.close
    else:
        return 0.0

def apply_stop_loss(row, SPREAD):
    if row.SIGNAL != NONE:
        if row.SIGNAL == BUY:
            # return row.low - SPREAD
            return row.open + SPREAD # ORGINAL
        else:
            # return row.high + SPREAD
            return row.open # ORGINAL
    else:
        return 0.0

def remove_spread(df):
    for a in ["ask", "bid"]:
        for b in ["o", "h", "l", "c"]:
            c = f"{a}_{b}"
            df[c] = df[f"mid_{b}"]

def apply_signals(df, PROFIT_FACTOR, sig,SPREAD):
    df["SIGNAL"] = df.apply(sig, axis=1)
    df["TP"] = df.apply(apply_take_profit, axis=1, PROFIT_FACTOR=PROFIT_FACTOR,SPREAD=SPREAD)
    df["SL"] = df.apply(apply_stop_loss, axis=1,SPREAD=SPREAD)

def create_signals(df, time_d=1):
    df_signals = df[df.SIGNAL != NONE].copy() 
    df_signals['m5_start'] = [x + dt.timedelta(hours=time_d) for x in df_signals.time]
    df_signals.drop(['time', 'open', 'high', 'low',
                'ENGULFING', 'EMA_200', 'RSI_14', 'direction'], axis=1, inplace=True)
    df_signals.rename(columns={
        'close' : 'start_price',
        'm5_start' : 'time'
    }, inplace=True)

    # print(df_signals)
    return df_signals


def create_signals2(df, time_d=1):
    df_signals = df[df.SIGNAL != NONE].copy() 
    df_signals['m5_start'] = [x + dt.timedelta(hours=time_d) for x in df_signals.time]
    df_signals.drop(['time', 'open', 'high', 'low'], axis=1, inplace=True)
    df_signals.rename(columns={
        'close' : 'start_price',
        'm5_start' : 'time'
    }, inplace=True)

    # print(df_signals)
    return df_signals


class Trade:
    def __init__(self, row, profit_factor, loss_factor, SPREAD):
        self.running = True
        self.start_index_m5 = row.name
        self.profit_factor = profit_factor
        self.loss_factor = loss_factor
        self.SPREAD = SPREAD
        
        if row.SIGNAL == BUY:
            self.start_price = row.start_price + self.SPREAD
            self.trigger_price = row.start_price + self.SPREAD
            
        if row.SIGNAL == SELL:
            self.start_price = row.start_price
            self.trigger_price = row.start_price
            
        self.SIGNAL = row.SIGNAL
        self.TP = row.TP
        self.SL = row.SL
        self.result = 0.0
        self.end_time = row.time
        self.start_time = row.time
        
    def close_trade(self, row, result, trigger_price):
        self.running = False
        self.result = result
        self.end_time = row.time
        self.trigger_price = trigger_price
        
    def update(self, row):
        if self.SIGNAL == BUY:
            if row.high >= self.TP:
                self.close_trade(row, self.profit_factor, row.high)
            elif row.low <= self.SL:
                self.close_trade(row, self.loss_factor, row.low)
        if self.SIGNAL == SELL:
            if row.low <= self.TP:
                self.close_trade(row, self.profit_factor, row.low + self.SPREAD)
            elif row.high >= self.SL:
                self.close_trade(row, self.loss_factor, row.high + self.SPREAD)   

class GuruTester2:
    def __init__(self, df_big,
                    df_m5,
                    SPREAD,
                    LOSS_FACTOR = -1.0,
                    PROFIT_FACTOR=2,
                    use_spread=True,
                    time_d=1 ):
        self.df_big = df_big.copy()
        self.SPREAD = SPREAD
        self.LOSS_FACTOR = LOSS_FACTOR
        self.PROFIT_FACTOR = PROFIT_FACTOR
        self.use_spread = use_spread
        self.df_m5 = df_m5.copy()
        self.time_d = time_d

        self.prepare_data()
        
    def prepare_data(self):
        
        print("prepare_data...")

        df_m5_slim = self.df_m5[['time', 'high', 'low']].copy()
        df_signals = create_signals2(self.df_big, time_d=self.time_d)

        self.merged = pd.merge(left=df_m5_slim, right=df_signals, on='time', how='left')
        self.merged.fillna(0, inplace=True)
        self.merged.SIGNAL = self.merged.SIGNAL.astype(int)

    def run_test(self):
        print("run_test...")
        open_trades_m5 = []
        closed_trades_m5 = []

        for index, row in self.merged.iterrows():
            
            if row.SIGNAL != NONE:
                open_trades_m5.append(Trade(row, self.PROFIT_FACTOR, self.LOSS_FACTOR, self.SPREAD))  
                
            for ot in open_trades_m5:
                ot.update(row)
                if ot.running == False:
                    closed_trades_m5.append(ot)
            open_trades_m5 = [x for x in open_trades_m5 if x.running == True]

        self.df_results = pd.DataFrame.from_dict([vars(x) for x in closed_trades_m5]) 
        win_rate = (self.df_results.result==self.PROFIT_FACTOR).sum()/len(self.df_results)
        print("Result:")
        print("win rate: ",win_rate)
        print(self.df_results.result.value_counts())
        # print("Result:", self.df_results.result.sum())
        return self.df_results

class GuruTester:
    def __init__(self, df_big,
                    apply_signal, 
                    df_m5,
                    SPREAD,
                    use_spread=True,
                    LOSS_FACTOR = -1.0,
                    PROFIT_FACTOR = 1.5,
                    time_d=1 ):
        self.df_big = df_big.copy()
        self.SPREAD = SPREAD
        self.use_spread = use_spread
        self.apply_signal = apply_signal
        self.df_m5 = df_m5.copy()
        self.LOSS_FACTOR = LOSS_FACTOR
        self.PROFIT_FACTOR = PROFIT_FACTOR
        self.time_d = time_d

        self.prepare_data()
        
    def prepare_data(self):
        
        print("prepare_data...")



        apply_signals(self.df_big,
                    self.PROFIT_FACTOR,
                    self.apply_signal,
                    self.SPREAD)


        df_m5_slim = self.df_m5[['time', 'high', 'low']].copy()
        df_signals = create_signals(self.df_big, time_d=self.time_d)

        self.merged = pd.merge(left=df_m5_slim, right=df_signals, on='time', how='left')
        self.merged.fillna(0, inplace=True)
        self.merged.SIGNAL = self.merged.SIGNAL.astype(int)

    def run_test(self):
        print("run_test...")
        open_trades_m5 = []
        closed_trades_m5 = []

        for index, row in self.merged.iterrows():
            
            if row.SIGNAL != NONE:
                open_trades_m5.append(Trade(row, self.PROFIT_FACTOR, self.LOSS_FACTOR, self.SPREAD))  
                
            for ot in open_trades_m5:
                ot.update(row)
                if ot.running == False:
                    closed_trades_m5.append(ot)
            open_trades_m5 = [x for x in open_trades_m5 if x.running == True]

        self.df_results = pd.DataFrame.from_dict([vars(x) for x in closed_trades_m5]) 
        print("Result:", self.df_results.result.sum())
        return self.df_results
