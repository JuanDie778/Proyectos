import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
import ta
def create_strategy_tab(df, symbol, metrics):
    """Crea la tab de estrategia y señales con velas japonesas y métricas avanzadas"""
    
    # Crear subplots para el gráfico principal y drawdown
    fig_main = go.Figure()
    
    # Velas japonesas
    fig_main.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Precio',
        increasing_line_color=colors['secondary'],
        decreasing_line_color='#FF6B6B',
        increasing_fillcolor=colors['secondary'],
        decreasing_fillcolor='#FF6B6B'
    ))
    
    # Medias móviles
    fig_main.add_trace(go.Scatter(x=df.index, y=df['SMA_15'], name='SMA 15', 
                            line=dict(color=colors['primary'], width=2)))
    fig_main.add_trace(go.Scatter(x=df.index, y=df['SMA_60'], name='SMA 60', 
                            line=dict(color=colors['accent'], width=2)))
    
    # Señales de compra y venta
    buy_signals = df.dropna(subset=['Buy_Signal'])
    sell_signals = df.dropna(subset=['Sell_Signal'])
    
    if not buy_signals.empty:
        fig_main.add_trace(go.Scatter(
            x=buy_signals.index, 
            y=buy_signals['Buy_Signal'], 
            mode='markers', 
            name='🟢 Comprar', 
            marker=dict(
                color='lime', 
                size=12, 
                symbol='triangle-up',
                line=dict(width=2, color='white')
            )
        ))
    
    if not sell_signals.empty:
        fig_main.add_trace(go.Scatter(
            x=sell_signals.index, 
            y=sell_signals['Sell_Signal'], 
            mode='markers', 
            name='🔴 Vender', 
            marker=dict(
                color='red', 
                size=12, 
                symbol='triangle-down',
                line=dict(width=2, color='white')
            )
        ))
    
    # Añadir área entre las SMAs para visualizar mejor el cruce
    fig_main.add_trace(go.Scatter(
        x=df.index, 
        y=df['SMA_15'], 
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig_main.add_trace(go.Scatter(
        x=df.index, 
        y=df['SMA_60'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='Zona SMA',
        fillcolor='rgba(102, 159, 238, 0.1)',
        showlegend=False
    ))
    
    fig_main.update_layout(
        title=f'🎯 Estrategia SMA Crossover con Velas Japonesas - {symbol}',
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        xaxis_gridcolor=colors['grid'],
        yaxis_gridcolor=colors['grid'],
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    # Configurar ejes
    fig_main.update_xaxes(gridcolor=colors['grid'])
    fig_main.update_yaxes(gridcolor=colors['grid'])
    
    # Gráfico de Drawdown separado
    fig_drawdown = go.Figure()
    
    fig_drawdown.add_trace(go.Scatter(
        x=df.index,
        y=df['Drawdown'],
        name='Drawdown',
        line=dict(color='#FF6B6B', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.3)'
    ))
    
    fig_drawdown.update_layout(
        title=f'📉 Análisis de Drawdown - {symbol}',
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        xaxis_gridcolor=colors['grid'],
        yaxis_gridcolor=colors['grid'],
        height=300,
        yaxis_title="Drawdown (%)"
    )
    
    fig_drawdown.update_xaxes(gridcolor=colors['grid'])
    fig_drawdown.update_yaxes(gridcolor=colors['grid'])
    
    # Estadísticas de señales más detalladas
    total_buy_signals = len(buy_signals)
    total_sell_signals = len(sell_signals)
    total_signals = total_buy_signals + total_sell_signals
    
    # Análisis del estado actual de la estrategia
    current_sma15 = df['SMA_15'].iloc[-1] if not df['SMA_15'].isna().iloc[-1] else 0
    current_sma60 = df['SMA_60'].iloc[-1] if not df['SMA_60'].isna().iloc[-1] else 0
    current_position = "ALCISTA" if current_sma15 > current_sma60 else "BAJISTA"
    position_color = colors['secondary'] if current_sma15 > current_sma60 else '#FF6B6B'
    
    # Calcular días desde última señal
    last_signal_date = None
    if not buy_signals.empty or not sell_signals.empty:
        all_signals = pd.concat([buy_signals[['Buy_Signal']], sell_signals[['Sell_Signal']]])
        last_signal_date = all_signals.index.max()
        days_since_signal = (df.index.max() - last_signal_date).days
    else:
        days_since_signal = "N/A"

    ## Función de estrategia simple (SMA Crossover)
def sma_crossover_strategy(df):
    """Estrategia de cruce de medias móviles"""
    df = df.copy()
    
    # Asegurar que tenemos los datos necesarios
    if 'SMA_15' not in df.columns or 'SMA_60' not in df.columns:
        return df
    
    # Eliminar valores NaN
    df = df.dropna(subset=['SMA_15', 'SMA_60'])
    
    if len(df) == 0:
        return df
    
    # Crear posiciones
    df['Position'] = 0
    df.loc[df['SMA_15'] > df['SMA_60'], 'Position'] = 1
    df.loc[df['SMA_15'] <= df['SMA_60'], 'Position'] = 0
    
    # Detectar cambios en posición
    df['Position_Change'] = df['Position'].diff()
    
    # Señales de compra y venta
    df['Buy_Signal'] = np.where(df['Position_Change'] == 1, df['Close'], np.nan)
    df['Sell_Signal'] = np.where(df['Position_Change'] == -1, df['Close'], np.nan)
    
    return df

# Indicadores avanzados para scalping
def calculate_advanced_scalping_indicators(df):
    """Calcula indicadores avanzados para scalping"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # EMAs rápidas para scalping
    df['EMA_8'] = ta.trend.ema_indicator(df['Close'], window=8)
    df['EMA_21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['EMA_55'] = ta.trend.ema_indicator(df['Close'], window=55)
    
    # RSI Estocástico
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
    df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14)
    
    # MACD para divergencias
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_Scalp'] = macd.macd()
    df['MACD_Signal_Scalp'] = macd.macd_signal()
    df['MACD_Hist_Scalp'] = macd.macd_diff()
    
    # VWAP (simulado con precio promedio ponderado)
    if 'Volume' in df.columns:
        cum_price_vol = (df['Close'] * df['Volume']).cumsum()
        cum_vol = df['Volume'].cumsum()
        # Avoid division by zero: only divide where cum_vol != 0
        df['VWAP'] = np.where(cum_vol != 0, cum_price_vol / cum_vol, df['Close'])
    else:
        df['VWAP'] = df['Close'].rolling(window=20).mean()
    
    # ATR para gestión de riesgo
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)

    # Parabolic SAR
    psar = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close'])
    df['SAR'] = psar.psar()
    
    # Williams %R
    df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
    
    # Detección de patrones de velas (simplificado)
    df['Candle_Body'] = abs(df['Close'] - df['Open'])
    df['Upper_Shadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Lower_Shadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']
    range_val = df['High'] - df['Low']
    df['Body_Ratio'] = np.where(range_val == 0, 0, df['Candle_Body'] / range_val)
    
    # Doji pattern (cuerpo pequeño)
    df['Is_Doji'] = df['Body_Ratio'] < 0.1
    
    # Hammer/Hanging Man (sombra inferior larga)
    df['Is_Hammer'] = (df['Lower_Shadow'] > 2 * df['Candle_Body']) & (df['Upper_Shadow'] < df['Candle_Body'])
    
    return df

# Estrategia avanzada de scalping
def advanced_scalping_strategy(df, risk_config):
    """Estrategia avanzada de scalping con múltiples indicadores"""
    df = df.copy()
    df = calculate_advanced_scalping_indicators(df)

    # Parámetros de la estrategia
    stop_loss_atr_mult = risk_config.get('stop_loss_atr', 1.5)
    take_profit_mult = risk_config.get('take_profit_ratio', 2.0)
    position_size_pct = risk_config.get('position_size', 0.02)  # 2% del capital por operación
    max_positions = risk_config.get('max_positions', 3)

    # Inicializar columnas
    df['Scalp_Position'] = 0
    df['Scalp_Score'] = 0
    df['Entry_Price'] = np.nan
    df['Stop_Loss'] = np.nan
    df['Take_Profit'] = np.nan
    df['Scalp_Buy_Signal'] = np.nan
    df['Scalp_Sell_Signal'] = np.nan
    df['Exit_Reason'] = ''

    # Filtros de tiempo (horarios de alta liquidez UTC)
    df['Hour'] = df.index.hour
    high_liquidity_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 8 AM - 9 PM UTC

    current_position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    positions_count = 0

    for i in range(len(df)):
        if i < 55:  # Necesitamos suficiente historial para EMAs
            continue

        # Obtener valores actuales
        close = df['Close'].iloc[i]
        ema_8 = df['EMA_8'].iloc[i]
        ema_21 = df['EMA_21'].iloc[i]
        ema_55 = df['EMA_55'].iloc[i]
        rsi = df['RSI_14'].iloc[i]
        stoch_k = df['Stoch_K'].iloc[i]
        stoch_d = df['Stoch_D'].iloc[i]
        macd_hist = df['MACD_Hist_Scalp'].iloc[i]
        vwap = df['VWAP'].iloc[i]
        atr = df['ATR'].iloc[i]
        if atr <= 0 or np.isnan(atr):
            atr = 0.0001  # Small positive value to avoid issues
        sar = df['SAR'].iloc[i]
        williams_r = df['Williams_R'].iloc[i]
        hour = df['Hour'].iloc[i]
        is_doji = df['Is_Doji'].iloc[i]
        is_hammer = df['Is_Hammer'].iloc[i]

        # Verificar horario de operación (solo para datos intradiarios)
        # Para datos diarios, permitir todas las horas
        if hour != 0 and hour not in high_liquidity_hours:
            continue
            
        # Calcular score de señal
        score = 0

        # Señales alcistas
        if (ema_8 > ema_21 > ema_55 and  # Tendencia alcista
            close > vwap and  # Precio sobre VWAP
            rsi > 20 and rsi < 80 and  # RSI en rango ampliado
            stoch_k > stoch_d and stoch_k > 20 and  # Estocástico alcista
            macd_hist > 0 and  # MACD positivo
            williams_r > -80 and  # Williams %R no oversold
            close > sar):  # Precio sobre SAR
            score += 1

            # Bonus por patrones de velas
            if is_hammer:
                score += 0.5
            if is_doji and i > 0 and df['Close'].iloc[i-1] < df['Open'].iloc[i-1]:  # Doji después de vela bajista
                score += 0.3

        # Señales bajistas
        elif (ema_8 < ema_21 < ema_55 and  # Tendencia bajista
              close < vwap and  # Precio bajo VWAP
              rsi > 20 and rsi < 80 and  # RSI en rango ampliado
              stoch_k < stoch_d and stoch_k < 80 and  # Estocástico bajista
              macd_hist < 0 and  # MACD negativo
              williams_r < -20 and  # Williams %R no overbought
              close < sar):  # Precio bajo SAR
            score -= 1

            # Bonus por patrones de velas
            if is_hammer:
                score -= 0.5
            if is_doji and i > 0 and df['Close'].iloc[i-1] > df['Open'].iloc[i-1]:  # Doji después de vela alcista
                score -= 0.3

        df.loc[df.index[i], 'Scalp_Score'] = score
        
        # Gestión de posiciones existentes
        if current_position != 0:
            # Verificar stop loss
            if (current_position == 1 and close <= stop_loss) or \
               (current_position == -1 and close >= stop_loss):
                df.loc[df.index[i], 'Scalp_Sell_Signal'] = close
                df.loc[df.index[i], 'Exit_Reason'] = 'Stop Loss'
                current_position = 0
                positions_count -= 1
                
            # Verificar take profit
            elif (current_position == 1 and close >= take_profit) or \
                 (current_position == -1 and close <= take_profit):
                df.loc[df.index[i], 'Scalp_Sell_Signal'] = close
                df.loc[df.index[i], 'Exit_Reason'] = 'Take Profit'
                current_position = 0
                positions_count -= 1
                
            # Salida por cambio de tendencia
            elif (current_position == 1 and score < -0.5) or \
                 (current_position == -1 and score > 0.5):
                df.loc[df.index[i], 'Scalp_Sell_Signal'] = close
                df.loc[df.index[i], 'Exit_Reason'] = 'Trend Change'
                current_position = 0
                positions_count -= 1
        
        # Nuevas entradas (solo si no hay posición activa y no excedemos max posiciones)
        elif positions_count < max_positions:
            # Señal de compra (score >= 1.0)
            if score >= 1.0:
                current_position = 1
                entry_price = close
                stop_loss = close - (atr * stop_loss_atr_mult)
                take_profit = close + (atr * stop_loss_atr_mult * take_profit_mult)
                positions_count += 1

                df.loc[df.index[i], 'Scalp_Buy_Signal'] = close
                df.loc[df.index[i], 'Entry_Price'] = entry_price
                df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                df.loc[df.index[i], 'Take_Profit'] = take_profit

            # Señal de venta (score <= -1.0)
            elif score <= -1.0:
                current_position = -1
                entry_price = close
                stop_loss = close + (atr * stop_loss_atr_mult)
                take_profit = close - (atr * stop_loss_atr_mult * take_profit_mult)
                positions_count += 1

                df.loc[df.index[i], 'Scalp_Sell_Signal'] = close  # En short, "sell" es la entrada
                df.loc[df.index[i], 'Entry_Price'] = entry_price
                df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                df.loc[df.index[i], 'Take_Profit'] = take_profit
        
        df.loc[df.index[i], 'Scalp_Position'] = current_position
    
    return df