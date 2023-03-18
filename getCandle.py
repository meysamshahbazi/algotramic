import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

class Rates:
    def __init__(self,symbol,number_of_data,timeframe):
        self.symbol = symbol
        self.number_of_data = number_of_data
        self.timeframe = timeframe
        mt5.initialize()

    def get_rates_from_now(self):
        # Compute now date
        from_date = datetime.now()
        # Extract n Ticks before now
        rates = mt5.copy_rates_from(self.symbol, self.timeframe, from_date, self.number_of_data)
        # Transform Tuple into a DataFrame
        df_rates = pd.DataFrame(rates)
        # Convert number format of the date into date format
        df_rates["time"] = pd.to_datetime(df_rates["time"], unit="s")
    #     df_rates = df_rates.set_index("time")
        return df_rates



