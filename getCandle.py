import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import datetime as dt

class Rates:
    def __init__(self,symbol,number_of_data,timeframe):
        self.symbol = symbol
        self.number_of_data = number_of_data
        self.timeframe = timeframe
        mt5.initialize()
        self.from_date = datetime.now() # - dt.timedelta(hours=-1)
        # Extract n Ticks before now
        ticks = mt5.copy_ticks_from(self.symbol, self.from_date, 10000, mt5.COPY_TICKS_ALL)
        # Transfrom a tuple into a dataframe
        df_ticks = pd.DataFrame(ticks)
        # print(df_ticks)
        # Convert number format of the date into date format
        df_ticks["time"] = pd.to_datetime(df_ticks["time"], unit="s")
        df_ticks['spread'] = df_ticks.ask-df_ticks.bid
        self.spread = df_ticks['spread'].max()

    def get_rates_from_now(self):
        # Compute now date
        # Extract n Ticks before now
        rates = mt5.copy_rates_from(self.symbol, self.timeframe, self.from_date, self.number_of_data)
        mt5.shutdown() 
        # Transform Tuple into a DataFrame
        df_rates = pd.DataFrame(rates)
        # Convert number format of the date into date format
        df_rates["time"] = pd.to_datetime(df_rates["time"], unit="s")
    #     df_rates = df_rates.set_index("time")
        return df_rates

    def get_spread(self):
        return self.spread




