import pandas as pd

def BollingerBands(df: pd.DataFrame, n=20, s=2):
    typical_p = ( df.close + df.high + df.low ) / 3
    stddev = typical_p.rolling(window=n).std()
    df['BB_MA'] = typical_p.rolling(window=n).mean()
    df['BB_UP'] = df['BB_MA'] + stddev * s
    df['BB_LW'] = df['BB_MA'] - stddev * s
    return df

def BollingerBandsFeature(df: pd.DataFrame, n=20, s=2):
    df = BollingerBands(df,n,s)
    df['Feat_BB_MA_c'] = df.BB_MA-df.close
    df['Feat_BB_UP_c'] = df.BB_UP-df.close
    df['Feat_BB_LW_c'] = df.BB_LW-df.close

    df['Feat_BB_MA_o'] = df.BB_MA-df.open
    df['Feat_BB_UP_o'] = df.BB_UP-df.open
    df['Feat_BB_LW_o'] = df.BB_LW-df.open

    df['Feat_BB_MA_l'] = df.BB_MA-df.low
    df['Feat_BB_UP_l'] = df.BB_UP-df.low
    df['Feat_BB_LW_l'] = df.BB_LW-df.low

    df['Feat_BB_MA_h'] = df.BB_MA-df.high
    df['Feat_BB_UP_h'] = df.BB_UP-df.high
    df['Feat_BB_LW_h'] = df.BB_LW-df.high
    return df


def ATR(df: pd.DataFrame, n=14):
    prev_c = df.close.shift(1)
    tr1 = df.high - df.low
    tr2 = df.high - prev_c
    tr3 = prev_c - df.low
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    df[f"ATR_{n}"] = tr.rolling(window=n).mean()
    return df

def ATRFeature(df: pd.DataFrame, n=14):
    prev_c = df.close.shift(1)
    tr1 = df.high - df.low
    tr2 = df.high - prev_c
    tr3 = prev_c - df.low
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    df[f"Feat_ATR_{n}"] = tr.rolling(window=n).mean()
    return df


def KeltnerChannels(df: pd.DataFrame, n_ema=20, n_atr=10):
    df['EMA'] = df.close.ewm(span=n_ema, min_periods=n_ema).mean()
    df = ATR(df, n=n_atr)
    c_atr = f"ATR_{n_atr}"
    df['KeUp'] = df[c_atr] * 2 + df.EMA
    df['KeLo'] = df.EMA - df[c_atr] * 2
    df.drop(c_atr, axis=1, inplace=True)
    return df

def KeltnerChannelsFeature(df: pd.DataFrame, n_ema=20, n_atr=10):
    df = KeltnerChannels(df,n_ema,n_atr)
    # df['EMA'] = df.close.ewm(span=n_ema, min_periods=n_ema).mean()
    df['Feat_EMA_c'] = df['EMA']-df.close
    df['Feat_KeUp_c'] = df['KeUp'] - df.close
    df['Feat_KeLo_c'] = df['KeLo'] - df.close

    df['Feat_EMA_o'] = df['EMA']-df.open
    df['Feat_KeUp_o'] = df['KeUp'] - df.open
    df['Feat_KeLo_o'] = df['KeLo'] - df.open

    df['Feat_EMA_h'] = df['EMA']-df.high
    df['Feat_KeUp_h'] = df['KeUp'] - df.high
    df['Feat_KeLo_h'] = df['KeLo'] - df.high

    df['Feat_EMA_l'] = df['EMA']-df.low
    df['Feat_KeUp_l'] = df['KeUp'] - df.low
    df['Feat_KeLo_l'] = df['KeLo'] - df.low
    return df


def RSI(df: pd.DataFrame, n=14):
    alpha = 1.0 / n
    gains = df.close.diff()

    wins = pd.Series([ x if x >= 0 else 0.0 for x in gains ], name="wins")
    losses = pd.Series([ x * -1 if x < 0 else 0.0 for x in gains ], name="losses")

    wins_rma = wins.ewm(min_periods=n, alpha=alpha).mean()
    losses_rma = losses.ewm(min_periods=n, alpha=alpha).mean()

    rs = wins_rma / losses_rma

    df[f"RSI_{n}"] = 100.0 - (100.0 / (1.0 + rs))
    return df

def RSIFeature(df: pd.DataFrame, n=14):
    alpha = 1.0 / n
    gains = df.close.diff()
    wins = pd.Series([ x if x >= 0 else 0.0 for x in gains ], name="wins")
    losses = pd.Series([ x * -1 if x < 0 else 0.0 for x in gains ], name="losses")
    df['Feat_gains'] = gains
    wins_rma = wins.ewm(min_periods=n, alpha=alpha).mean()
    losses_rma = losses.ewm(min_periods=n, alpha=alpha).mean()
    df['Feat_wins_rma'] = wins_rma
    df['Feat_losses_rma'] = losses_rma

    rs = wins_rma / losses_rma
    # df['Feat_rs'] = rs
    df[f"Feat_RSI_{n}"] = 1.0 - (1.0 / (1.0 + rs))
    return df


def MACD(df: pd.DataFrame, n_slow=26, n_fast=12, n_signal=9):

    ema_long = df.close.ewm(min_periods=n_slow, span=n_slow).mean()
    ema_short = df.close.ewm(min_periods=n_fast, span=n_fast).mean()

    df['MACD'] = ema_short - ema_long
    df['SIGNAL_MACD'] = df.MACD.ewm(min_periods=n_signal, span=n_signal).mean()
    df['HIST'] = df.MACD - df.SIGNAL_MACD

    return df

def MACDFeature(df: pd.DataFrame, n_slow=26, n_fast=12, n_signal=9):
    df = MACD(df,n_slow,n_fast,n_signal)
    df['Feat_MACD']  = df['MACD'] 
    df['Feat_SIGNAL_MACD'] = df['SIGNAL_MACD']
    df['Feat_HIST'] = df['HIST'] 
    return df





























