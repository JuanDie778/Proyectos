def Frogames_strategy(df, duration=5,spread=0):
  """EL DATAFRAME NECESITA TENER los siguientes nombres de columna: alta, baja, cierre"""

  # Support and resistance building
  df["support"] = np.nan
  df["resistance"] = np.nan

  df.loc[(df["low"].shift(5) > df["low"].shift(4)) &
        (df["low"].shift(4) > df["low"].shift(3)) &
        (df["low"].shift(3) > df["low"].shift(2)) &
        (df["low"].shift(2) > df["low"].shift(1)) &
        (df["low"].shift(1) > df["low"].shift(0)), "support"] = df["low"]


  df.loc[(df["high"].shift(5) < df["high"].shift(4)) &
  (df["high"].shift(4) < df["high"].shift(3)) &
  (df["high"].shift(3) < df["high"].shift(2)) &
  (df["high"].shift(2) < df["high"].shift(1)) &
  (df["high"].shift(1) < df["high"].shift(0)), "resistance"] = df["high"]


  # Create Simple moving average 30 days
  df["SMA fast"] = df["close"].rolling(30).mean()

  # Create Simple moving average 60 days
  df["SMA slow"] = df["close"].rolling(60).mean()

  df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=10).rsi()

  # RSI yersteday
  df["rsi yersteday"] = df["rsi"].shift(1)

  # Create the signal
  df["signal"] = 0

  df["smooth resistance"] = df["resistance"].fillna(method="ffill")
  df["smooth support"] = df["support"].fillna(method="ffill")


  condition_1_buy = (df["close"].shift(1) < df["smooth resistance"].shift(1)) & \
                    (df["smooth resistance"]*(1+0.5/100) < df["close"])
  condition_2_buy = df["SMA fast"] > df["SMA slow"]

  condition_3_buy = df["rsi"] < df["rsi yersteday"]

  condition_1_sell = (df["close"].shift(1) > df["smooth support"].shift(1)) & \
                    (df["smooth support"]*(1+0.5/100) > df["close"])
  condition_2_sell = df["SMA fast"] < df["SMA slow"]

  condition_3_sell = df["rsi"] > df["rsi yersteday"]



  df.loc[condition_1_buy & condition_2_buy & condition_3_buy, "signal"] = 1
  df.loc[condition_1_sell & condition_2_sell & condition_3_sell, "signal"] = -1


  # Calculamos las ganancias
  df["pct"] = df["close"].pct_change(1)

  df["return"] = np.array([df["pct"].shift(i) for i in range(duration)]).sum(axis=0) * (df["signal"].shift(duration))
  df.loc[df["return"]==-1, "return"] = df["return"]-spread
  df.loc[df["return"]==1, "return"] = df["return"]-spread


  return df["return"]