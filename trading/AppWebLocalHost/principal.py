import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Import advanced scalping strategy
from Estrategia import advanced_scalping_strategy, calculate_advanced_scalping_indicators
# Configuración de la aplicación
app = dash.Dash(__name__)
app.title = "Trading Dashboard - Análisis Algorítmico"

# Estilos CSS personalizados
colors = {
    'background': '#1E1E1E',
    'text': '#C9C9C9',
    'primary': '#669FEE',
    'secondary': '#66EE91',
    'accent': '#EECC55',
    'grid': '#474A4A'
}

# Función para obtener datos
def get_crypto_data(symbol="BTC-USD", period="1y"):
    """Obtiene datos históricos de criptomonedas"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        df.index.name = 'Date'
        return df
    except Exception as e:
        print(f"Error obteniendo datos: {e}")
        return pd.DataFrame()

def get_benchmark_data(symbol="^GSPC", period="1y"):
    """Obtiene datos del benchmark (S&P 500)"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        return df['Close']
    except Exception as e:
        print(f"Error obteniendo benchmark: {e}")
        return pd.Series()

# Función para calcular indicadores técnicos
def calculate_indicators(df):
    """Calcula diversos indicadores técnicos"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Medias móviles
    df['SMA_15'] = ta.trend.sma_indicator(df['Close'], window=15)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=7)
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_middle'] = bollinger.bollinger_mavg()
    df['BB_lower'] = bollinger.bollinger_lband()
    
    # Media móvil simple del volumen (calculada manualmente)
    if 'Volume' in df.columns:
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    return df

# Función de estrategia simple (SMA Crossover)
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
# Función de estrategia Frogames
def frogames_strategy(df):
    """Estrategia Frogames basada en soporte/resistencia, SMAs y RSI"""
    df = df.copy()

    # Construcción de soporte y resistencia
    df["support"] = np.nan
    df["resistance"] = np.nan

    df.loc[(df["Low"].shift(5) > df["Low"].shift(4)) &
        (df["Low"].shift(4) > df["Low"].shift(3)) &
        (df["Low"].shift(3) > df["Low"].shift(2)) &
        (df["Low"].shift(2) > df["Low"].shift(1)) &
        (df["Low"].shift(1) > df["Low"].shift(0)), "support"] = df["Low"]

    df.loc[(df["High"].shift(5) < df["High"].shift(4)) &
    (df["High"].shift(4) < df["High"].shift(3)) &
    (df["High"].shift(3) < df["High"].shift(2)) &
    (df["High"].shift(2) < df["High"].shift(1)) &
    (df["High"].shift(1) < df["High"].shift(0)), "resistance"] = df["High"]

    # Crear medias móviles simples de 30 y 60 días
    df["SMA fast"] = df["Close"].rolling(30).mean()
    df["SMA slow"] = df["Close"].rolling(60).mean()

    # RSI de 10 períodos
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=10).rsi()

    # RSI del día anterior
    df["rsi yesterday"] = df["rsi"].shift(1)

    # Crear las señales
    df["signal"] = 0

    df["smooth_resistance"] = df["resistance"].fillna(method="ffill")
    df["smooth_support"] = df["support"].fillna(method="ffill")

    condition_1_buy = (df["Close"].shift(1) < df["smooth_resistance"].shift(1)) & \
                    (df["smooth_resistance"]*(1+2/100) < df["Close"])
    condition_2_buy = df["SMA fast"] > df["SMA slow"]
    condition_3_buy = df["rsi"] < df["rsi yesterday"]

    condition_1_sell = (df["Close"].shift(1) > df["smooth_support"].shift(1)) & \
                    (df["smooth_support"]*(1+2/100) > df["Close"])
    condition_2_sell = df["SMA fast"] < df["SMA slow"]
    condition_3_sell = df["rsi"] > df["rsi yesterday"]

    # More flexible conditions: require breakout + trend, optional RSI
    buy_combined = condition_1_buy & condition_2_buy
    sell_combined = condition_1_sell & condition_2_sell

    df.loc[buy_combined, "signal"] = 1
    df.loc[sell_combined, "signal"] = -1

    # Establecer Position como señales forward-filled
    df['Position'] = df['signal'].fillna(method='ffill').fillna(0)

    # Columnas de señales para visualización
    df['Frogames_Buy_Signal'] = np.where(df['signal'] == 1, df['Close'], np.nan)
    df['Frogames_Sell_Signal'] = np.where(df['signal'] == -1, df['Close'], np.nan)

    return df

# Función para backtesting con métricas avanzadas

# Función para backtesting con métricas avanzadas
def backtest_strategy(df, benchmark_data, initial_capital=10000, risk_free_rate=0.02):
    """Realiza backtesting de la estrategia con métricas avanzadas"""
    df = df.copy()
    
    # Asegurar que tenemos la columna Position
    if 'Position' not in df.columns:
        df['Position'] = 0
    
    # Calcular retornos
    df['Returns'] = df['Close'].pct_change()
    df['Returns'] = df['Returns'].fillna(0).replace([np.inf, -np.inf], 0)
    df['Strategy_Returns'] = df['Returns'] * df['Position'].shift(1)

    # Reemplazar valores NaN e infinitos
    df['Strategy_Returns'] = df['Strategy_Returns'].fillna(0)
    df['Strategy_Returns'] = df['Strategy_Returns'].replace([np.inf, -np.inf], 0)
    
    # Calcular retornos acumulativos
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Portfolio_Value'] = initial_capital * df['Cumulative_Returns']
    
    # Calcular drawdown
    peak = df['Portfolio_Value'].expanding().max()
    df['Drawdown'] = (df['Portfolio_Value'] / peak - 1) * 100
    
    # Sincronizar datos del benchmark con nuestro DataFrame
    benchmark_aligned = benchmark_data.reindex(df.index, method='ffill')
    df['Benchmark_Returns'] = benchmark_aligned.pct_change().fillna(0)
    
    # Calcular métricas avanzadas
    metrics = calculate_advanced_metrics(df, risk_free_rate)
    
    return df, metrics

def calculate_advanced_metrics(df, risk_free_rate=0.02):
    """Calcula métricas avanzadas de rendimiento"""
    metrics = {}

    # Retornos básicos
    strategy_returns = df['Strategy_Returns'].dropna()
    benchmark_returns = df['Benchmark_Returns'].dropna()

    if len(strategy_returns) == 0:
        base_metrics = {key: 0 for key in ['total_return', 'annual_return', 'volatility', 'sharpe_ratio',
                                  'sortino_ratio', 'max_drawdown', 'alpha', 'beta', 'calmar_ratio']}
        scalping_metrics = {key: 0 for key in ['avg_trade_duration', 'win_loss_ratio', 'max_consecutive_losses', 'profit_factor']}
        metrics = {**base_metrics, **scalping_metrics}
        return metrics

    # Métricas de rendimiento
    final_value = df['Portfolio_Value'].iloc[-1]
    initial_value = df['Portfolio_Value'].iloc[0]

    metrics['total_return'] = (final_value / initial_value - 1) * 100

    # Retorno anualizado
    years = len(df) / 252  # Asumiendo días de trading
    metrics['annual_return'] = ((final_value / initial_value) ** (1/years) - 1) * 100 if years > 0 else 0

    # Volatilidad anualizada
    metrics['volatility'] = strategy_returns.std() * np.sqrt(252) * 100

    # Sharpe Ratio
    excess_return = metrics['annual_return'] - risk_free_rate * 100
    metrics['sharpe_ratio'] = (excess_return / metrics['volatility']) if metrics['volatility'] != 0 else 0

    # Sortino Ratio (solo considera desviación negativa)
    negative_returns = strategy_returns[strategy_returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252) * 100 if len(negative_returns) > 0 else 0.001
    metrics['sortino_ratio'] = (excess_return / downside_deviation) if downside_deviation != 0 else 0

    # Max Drawdown
    metrics['max_drawdown'] = df['Drawdown'].min()

    # Alpha y Beta
    if len(benchmark_returns) > 30 and len(strategy_returns) > 30:  # Mínimo 30 observaciones
        # Alinear retornos para el cálculo
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_idx) > 30:
            strat_aligned = strategy_returns.loc[common_idx]
            bench_aligned = benchmark_returns.loc[common_idx]

            # Beta (covarianza / varianza del benchmark)
            covariance = np.cov(strat_aligned, bench_aligned)[0, 1]
            benchmark_variance = np.var(bench_aligned)
            metrics['beta'] = covariance / benchmark_variance if benchmark_variance != 0 else 0

            # Alpha (retorno estrategia - (risk_free + beta * (retorno_benchmark - risk_free)))
            benchmark_annual_return = (1 + bench_aligned.mean()) ** 252 - 1
            alpha_calculation = (metrics['annual_return']/100) - (risk_free_rate + metrics['beta'] * (benchmark_annual_return - risk_free_rate))
            metrics['alpha'] = alpha_calculation * 100
        else:
            metrics['beta'] = 0
            metrics['alpha'] = 0
    else:
        metrics['beta'] = 0
        metrics['alpha'] = 0

    # Calmar Ratio (retorno anualizado / max drawdown absoluto)
    abs_max_drawdown = abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0.001
    metrics['calmar_ratio'] = metrics['annual_return'] / abs_max_drawdown if abs_max_drawdown != 0 else 0

    # Métricas específicas de scalping
    if 'Scalp_Buy_Signal' in df.columns and 'Scalp_Sell_Signal' in df.columns:
        # Calcular trades individuales
        buy_signals = df.dropna(subset=['Scalp_Buy_Signal'])
        sell_signals = df.dropna(subset=['Scalp_Sell_Signal'])

        if not buy_signals.empty and not sell_signals.empty:
            # Emparejar buys y sells (simplificado: asumir orden cronológico)
            all_signals = []
            for idx in buy_signals.index:
                all_signals.append(('buy', idx, df.loc[idx, 'Scalp_Buy_Signal']))
            for idx in sell_signals.index:
                all_signals.append(('sell', idx, df.loc[idx, 'Scalp_Sell_Signal']))

            all_signals.sort(key=lambda x: x[1])  # Ordenar por tiempo

            trades = []
            open_trade = None
            for signal_type, idx, price in all_signals:
                if signal_type == 'buy' and open_trade is None:
                    open_trade = {'entry': idx, 'entry_price': price}
                elif signal_type == 'sell' and open_trade is not None:
                    open_trade['exit'] = idx
                    open_trade['exit_price'] = price
                    open_trade['duration'] = (idx - open_trade['entry']).days
                    open_trade['pnl'] = (price - open_trade['entry_price']) / open_trade['entry_price'] * 100
                    trades.append(open_trade)
                    open_trade = None

            if trades:
                # Average trade duration
                metrics['avg_trade_duration'] = np.mean([t['duration'] for t in trades])

                # Win/Loss ratio
                winning_trades = [t for t in trades if t['pnl'] > 0]
                losing_trades = [t for t in trades if t['pnl'] <= 0]
                metrics['win_loss_ratio'] = len(winning_trades) / len(losing_trades) if losing_trades else float('inf')

                # Max consecutive losses
                consecutive_losses = 0
                max_consecutive = 0
                for trade in trades:
                    if trade['pnl'] <= 0:
                        consecutive_losses += 1
                        max_consecutive = max(max_consecutive, consecutive_losses)
                    else:
                        consecutive_losses = 0
                metrics['max_consecutive_losses'] = max_consecutive

                # Profit factor
                total_wins = sum(t['pnl'] for t in winning_trades)
                total_losses = abs(sum(t['pnl'] for t in losing_trades))
                metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf')
            else:
                metrics['avg_trade_duration'] = 0
                metrics['win_loss_ratio'] = 0
                metrics['max_consecutive_losses'] = 0
                metrics['profit_factor'] = 0
        else:
            metrics['avg_trade_duration'] = 0
            metrics['win_loss_ratio'] = 0
            metrics['max_consecutive_losses'] = 0
            metrics['profit_factor'] = 0
    else:
        metrics['avg_trade_duration'] = 0
        metrics['win_loss_ratio'] = 0
        metrics['max_consecutive_losses'] = 0
        metrics['profit_factor'] = 0

    return metrics

class MarketAnalyzer:
    """
    Analizador de mercado multi-temporal que evalúa tendencias, momentum,
    volatilidad y patrones de velas en diferentes horizontes temporales.
    """

    def __init__(self, df):
        """
        Inicializa el analizador con un DataFrame de precios OHLCV
        """
        self.df = df.copy()
        self.analysis_results = {}

    def calculate_advanced_indicators(self):
        """
        Calcula indicadores técnicos avanzados adicionales
        """
        df = self.df

        # Medias Móviles adicionales
        df['MA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['MA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['MA_200'] = ta.trend.sma_indicator(df['Close'], window=200)

        # EMA para análisis de momentum
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)

        # MACD (ya existe pero asegurar cálculo)
        if 'MACD' not in df.columns:
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_hist'] = macd.macd_diff()

        # RSI (ya existe)
        if 'RSI' not in df.columns:
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        # Estocástico
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()

        # Bollinger Bands (ya existe pero asegurar)
        if 'BB_upper' not in df.columns:
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_middle'] = bollinger.bollinger_mavg()
            df['BB_lower'] = bollinger.bollinger_lband()

        # Bandas de Bollinger - Ancho y posición
        df['BB_width'] = ((df['BB_upper'] - df['BB_lower']) / df['BB_middle']) * 100
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # Ichimoku Cloud (simplificado)
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        df['Ichimoku_Tenkan'] = (high_9 + low_9) / 2

        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        df['Ichimoku_Kijun'] = (high_26 + low_26) / 2

        df['Ichimoku_Senkou_A'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)

        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()
        df['Ichimoku_Senkou_B'] = ((high_52 + low_52) / 2).shift(26)

        # On-Balance Volume (OBV)
        if 'Volume' in df.columns:
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

        # Average True Range (ATR)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

        # Puntos Pivote
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
        df['S2'] = df['Pivot'] - (df['High'] - df['Low'])

        self.df = df
        return df

    def detect_candlestick_patterns(self):
        """
        Detecta patrones de velas japonesas principales
        """
        df = self.df
        patterns = {}

        # Calcular variables auxiliares
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Upper_Shadow'] = df['High'] - np.maximum(df['Close'], df['Open'])
        df['Lower_Shadow'] = np.minimum(df['Close'], df['Open']) - df['Low']
        df['Body_Size'] = df['Body'] / (df['High'] - df['Low'])

        # Definir tamaños relativos
        small_body_threshold = 0.3
        large_body_threshold = 0.7

        # PATRONES DE REVERSIÓN

        # Martillo (Hammer)
        hammer_condition = (
            (df['Lower_Shadow'] >= 2 * df['Body']) &
            (df['Upper_Shadow'] <= 0.1 * df['Body']) &
            (df['Body_Size'] <= small_body_threshold)
        )
        patterns['Hammer'] = hammer_condition

        # Hombre Colgado (Hanging Man)
        hanging_man_condition = (
            (df['Lower_Shadow'] >= 2 * df['Body']) &
            (df['Upper_Shadow'] <= 0.1 * df['Body']) &
            (df['Body_Size'] <= small_body_threshold) &
            (df['Close'].shift(1) > df['Close'].shift(2))  # En tendencia alcista previa
        )
        patterns['Hanging_Man'] = hanging_man_condition

        # Engulfing Alcista
        bullish_engulfing = (
            (df['Open'] < df['Close']) &  # Vela actual alcista
            (df['Open'].shift(1) > df['Close'].shift(1)) &  # Vela anterior bajista
            (df['Close'] > df['Open'].shift(1)) &  # Cierre actual > apertura anterior
            (df['Open'] < df['Close'].shift(1))  # Apertura actual < cierre anterior
        )
        patterns['Bullish_Engulfing'] = bullish_engulfing

        # Engulfing Bajista
        bearish_engulfing = (
            (df['Open'] > df['Close']) &  # Vela actual bajista
            (df['Open'].shift(1) < df['Close'].shift(1)) &  # Vela anterior alcista
            (df['Close'] < df['Open'].shift(1)) &  # Cierre actual < apertura anterior
            (df['Open'] > df['Close'].shift(1))  # Apertura actual > cierre anterior
        )
        patterns['Bearish_Engulfing'] = bearish_engulfing

        # Harami Alcista
        bullish_harami = (
            (df['Open'].shift(1) > df['Close'].shift(1)) &  # Vela anterior bajista
            (df['Open'] < df['Close']) &  # Vela actual alcista
            (df['Open'] > df['Close'].shift(1)) &  # Apertura actual > cierre anterior
            (df['Close'] < df['Open'].shift(1))  # Cierre actual < apertura anterior
        )
        patterns['Bullish_Harami'] = bullish_harami

        # Harami Bajista
        bearish_harami = (
            (df['Open'].shift(1) < df['Close'].shift(1)) &  # Vela anterior alcista
            (df['Open'] > df['Close']) &  # Vela actual bajista
            (df['Open'] < df['Close'].shift(1)) &  # Apertura actual < cierre anterior
            (df['Close'] > df['Open'].shift(1))  # Cierre actual > apertura anterior
        )
        patterns['Bearish_Harami'] = bearish_harami

        # PATRONES DE CONTINUACIÓN

        # Marubozu Alcista
        bullish_marubozu = (
            (df['Body_Size'] >= large_body_threshold) &
            (df['Upper_Shadow'] <= 0.05 * df['Body']) &
            (df['Lower_Shadow'] <= 0.05 * df['Body']) &
            (df['Close'] > df['Open'])
        )
        patterns['Bullish_Marubozu'] = bullish_marubozu

        # Marubozu Bajista
        bearish_marubozu = (
            (df['Body_Size'] >= large_body_threshold) &
            (df['Upper_Shadow'] <= 0.05 * df['Body']) &
            (df['Lower_Shadow'] <= 0.05 * df['Body']) &
            (df['Close'] < df['Open'])
        )
        patterns['Bearish_Marubozu'] = bearish_marubozu

        # Agregar patrones al DataFrame
        for pattern_name, pattern_condition in patterns.items():
            df[pattern_name] = pattern_condition

        self.df = df
        self.patterns = patterns
        return patterns

    def analyze_timeframe(self, timeframe_type):
        """
        Analiza el mercado según el horizonte temporal especificado

        timeframe_type: 'short' (1-5 días), 'medium' (5-20 días), 'long' (20+ días)
        """
        df = self.df
        analysis = {}

        if timeframe_type == 'short':
            # Análisis de corto plazo (últimos 5 días)
            recent_data = df.tail(5)
            analysis['period_name'] = 'Corto Plazo (1-5 días)'
            analysis['focus'] = 'Condiciones de sobrecompra/sobreventa y patrones recientes'

        elif timeframe_type == 'medium':
            # Análisis de mediano plazo (últimos 20 días)
            recent_data = df.tail(20)
            analysis['period_name'] = 'Mediano Plazo (5-20 días)'
            analysis['focus'] = 'Momentum y tendencia intermedia'

        elif timeframe_type == 'long':
            # Análisis de largo plazo (últimos 60 días o disponibles)
            recent_data = df.tail(min(60, len(df)))
            analysis['period_name'] = 'Largo Plazo (20+ días)'
            analysis['focus'] = 'Tendencia principal y estructura del mercado'

        current_price = df['Close'].iloc[-1]

        # 1. ANÁLISIS DE TENDENCIA
        trend_analysis = {}

        if timeframe_type == 'long':
            # Usar MA200 para largo plazo
            if not pd.isna(df['MA_200'].iloc[-1]):
                if current_price > df['MA_200'].iloc[-1]:
                    trend_analysis['primary_trend'] = 'Alcista'
                    trend_analysis['trend_strength'] = abs(current_price - df['MA_200'].iloc[-1]) / df['MA_200'].iloc[-1] * 100
                else:
                    trend_analysis['primary_trend'] = 'Bajista'
                    trend_analysis['trend_strength'] = abs(current_price - df['MA_200'].iloc[-1]) / df['MA_200'].iloc[-1] * 100
            else:
                trend_analysis['primary_trend'] = 'Sin datos suficientes'
                trend_analysis['trend_strength'] = 0

        elif timeframe_type == 'medium':
            # Usar MA50 para mediano plazo
            if not pd.isna(df['MA_50'].iloc[-1]):
                if current_price > df['MA_50'].iloc[-1]:
                    trend_analysis['primary_trend'] = 'Alcista'
                    trend_analysis['trend_strength'] = abs(current_price - df['MA_50'].iloc[-1]) / df['MA_50'].iloc[-1] * 100
                else:
                    trend_analysis['primary_trend'] = 'Bajista'
                    trend_analysis['trend_strength'] = abs(current_price - df['MA_50'].iloc[-1]) / df['MA_50'].iloc[-1] * 100
            else:
                trend_analysis['primary_trend'] = 'Sin datos suficientes'
                trend_analysis['trend_strength'] = 0

        else:  # short
            # Usar MA20 para corto plazo
            if not pd.isna(df['MA_20'].iloc[-1]):
                if current_price > df['MA_20'].iloc[-1]:
                    trend_analysis['primary_trend'] = 'Alcista'
                    trend_analysis['trend_strength'] = abs(current_price - df['MA_20'].iloc[-1]) / df['MA_20'].iloc[-1] * 100
                else:
                    trend_analysis['primary_trend'] = 'Bajista'
                    trend_analysis['trend_strength'] = abs(current_price - df['MA_20'].iloc[-1]) / df['MA_20'].iloc[-1] * 100
            else:
                trend_analysis['primary_trend'] = 'Sin datos suficientes'
                trend_analysis['trend_strength'] = 0

        # MACD para confirmación de tendencia
        macd_current = df['MACD'].iloc[-1]
        macd_signal = df['MACD_signal'].iloc[-1]

        if macd_current > macd_signal:
            trend_analysis['macd_direction'] = 'Alcista'
        else:
            trend_analysis['macd_direction'] = 'Bajista'

        # 2. ANÁLISIS DE MOMENTUM
        momentum_analysis = {}

        current_rsi = df['RSI'].iloc[-1]
        if current_rsi >= 70:
            momentum_analysis['rsi_condition'] = 'Sobrecomprado'
            momentum_analysis['rsi_signal'] = 'Posible corrección bajista'
        elif current_rsi <= 30:
            momentum_analysis['rsi_condition'] = 'Sobrevendido'
            momentum_analysis['rsi_signal'] = 'Posible rebote alcista'
        else:
            momentum_analysis['rsi_condition'] = 'Neutral'
            momentum_analysis['rsi_signal'] = 'Sin señal extrema'

        momentum_analysis['rsi_value'] = current_rsi

        # Estocástico
        current_stoch_k = df['Stoch_K'].iloc[-1]
        current_stoch_d = df['Stoch_D'].iloc[-1]

        if current_stoch_k >= 80:
            momentum_analysis['stoch_condition'] = 'Sobrecomprado'
        elif current_stoch_k <= 20:
            momentum_analysis['stoch_condition'] = 'Sobrevendido'
        else:
            momentum_analysis['stoch_condition'] = 'Neutral'

        # 3. ANÁLISIS DE VOLATILIDAD
        volatility_analysis = {}

        # Bollinger Bands
        bb_position = df['BB_position'].iloc[-1]
        bb_width = df['BB_width'].iloc[-1]

        if bb_position >= 0.8:
            volatility_analysis['bb_position'] = 'Cerca del límite superior'
            volatility_analysis['bb_signal'] = 'Posible resistencia'
        elif bb_position <= 0.2:
            volatility_analysis['bb_position'] = 'Cerca del límite inferior'
            volatility_analysis['bb_signal'] = 'Posible soporte'
        else:
            volatility_analysis['bb_position'] = 'Zona media'
            volatility_analysis['bb_signal'] = 'Sin señal extrema'

        # Clasificar volatilidad por ancho de las bandas
        bb_width_avg = df['BB_width'].tail(20).mean()
        if bb_width > bb_width_avg * 1.2:
            volatility_analysis['volatility_level'] = 'Alta'
        elif bb_width < bb_width_avg * 0.8:
            volatility_analysis['volatility_level'] = 'Baja'
        else:
            volatility_analysis['volatility_level'] = 'Normal'

        # 4. ANÁLISIS DE VOLUMEN
        volume_analysis = {}

        if 'Volume' in df.columns and 'OBV' in df.columns:
            current_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].tail(20).mean()

            if current_volume > avg_volume * 1.5:
                volume_analysis['volume_level'] = 'Alto'
                volume_analysis['volume_signal'] = 'Fuerte participación del mercado'
            elif current_volume < avg_volume * 0.5:
                volume_analysis['volume_level'] = 'Bajo'
                volume_analysis['volume_signal'] = 'Débil participación del mercado'
            else:
                volume_analysis['volume_level'] = 'Normal'
                volume_analysis['volume_signal'] = 'Participación típica del mercado'

            # Tendencia del OBV
            obv_trend = df['OBV'].tail(5).diff().sum()
            if obv_trend > 0:
                volume_analysis['obv_trend'] = 'Alcista'
            else:
                volume_analysis['obv_trend'] = 'Bajista'
        else:
            volume_analysis = {'volume_level': 'Sin datos', 'volume_signal': 'Sin datos de volumen'}

        # 5. PATRONES DE VELAS
        candlestick_analysis = {}
        recent_patterns = []

        # Buscar patrones en las últimas 3 velas
        for pattern_name, pattern_series in self.patterns.items():
            if pattern_series.tail(3).any():
                recent_patterns.append(pattern_name)

        candlestick_analysis['detected_patterns'] = recent_patterns

        # 6. NIVELES DE SOPORTE Y RESISTENCIA
        support_resistance = {}

        # Usar puntos pivote
        current_pivot = df['Pivot'].iloc[-1]
        current_r1 = df['R1'].iloc[-1]
        current_s1 = df['S1'].iloc[-1]

        support_resistance['pivot'] = current_pivot
        support_resistance['resistance_1'] = current_r1
        support_resistance['support_1'] = current_s1

        # Distancia a niveles clave
        distance_to_r1 = (current_r1 - current_price) / current_price * 100
        distance_to_s1 = (current_price - current_s1) / current_price * 100

        support_resistance['distance_to_resistance'] = distance_to_r1
        support_resistance['distance_to_support'] = distance_to_s1

        # Compilar análisis
        analysis.update({
            'trend': trend_analysis,
            'momentum': momentum_analysis,
            'volatility': volatility_analysis,
            'volume': volume_analysis,
            'candlesticks': candlestick_analysis,
            'support_resistance': support_resistance,
            'current_price': current_price,
            'timestamp': datetime.now()
        })

        return analysis

    def generate_summary(self, timeframe_type):
        """
        Genera un resumen en lenguaje sencillo del análisis
        """
        analysis = self.analyze_timeframe(timeframe_type)

        summary_parts = []

        # Introducción
        summary_parts.append(f"📊 **{analysis['period_name']}**")
        summary_parts.append(f"Precio actual: ${analysis['current_price']:.2f}")
        summary_parts.append("")

        # Tendencia
        trend = analysis['trend']
        trend_emoji = "📈" if trend['primary_trend'] == 'Alcista' else "📉" if trend['primary_trend'] == 'Bajista' else "➡️"
        summary_parts.append(f"{trend_emoji} **Tendencia Principal:** {trend['primary_trend']}")

        if trend['trend_strength'] > 0:
            strength_desc = "fuerte" if trend['trend_strength'] > 5 else "moderada" if trend['trend_strength'] > 2 else "débil"
            summary_parts.append(f"La tendencia es {strength_desc} con una separación del {trend['trend_strength']:.1f}% respecto a la media móvil de referencia.")

        # MACD confirmación
        macd_emoji = "✅" if trend['macd_direction'] == trend['primary_trend'] else "⚠️"
        summary_parts.append(f"{macd_emoji} MACD confirma tendencia {trend['macd_direction'].lower()}")
        summary_parts.append("")

        # Momentum
        momentum = analysis['momentum']
        rsi_emoji = "🔴" if momentum['rsi_condition'] == 'Sobrecomprado' else "🟢" if momentum['rsi_condition'] == 'Sobrevendido' else "🟡"
        summary_parts.append(f"{rsi_emoji} **Momentum (RSI {momentum['rsi_value']:.0f}):** {momentum['rsi_condition']}")
        summary_parts.append(f"{momentum['rsi_signal']}")
        summary_parts.append("")

        # Volatilidad
        volatility = analysis['volatility']
        vol_emoji = "🌪️" if volatility['volatility_level'] == 'Alta' else "😴" if volatility['volatility_level'] == 'Baja' else "🌊"
        summary_parts.append(f"{vol_emoji} **Volatilidad:** {volatility['volatility_level']}")
        summary_parts.append(f"El precio se encuentra {volatility['bb_position']} de las Bollinger Bands. {volatility['bb_signal']}")
        summary_parts.append("")

        # Volumen
        volume = analysis['volume']
        if 'volume_level' in volume and volume['volume_level'] != 'Sin datos':
            vol_emoji = "📊" if volume['volume_level'] == 'Alto' else "📉" if volume['volume_level'] == 'Bajo' else "📈"
            summary_parts.append(f"{vol_emoji} **Volumen:** {volume['volume_level']}")
            summary_parts.append(f"{volume['volume_signal']}")
            if 'obv_trend' in volume:
                summary_parts.append(f"Tendencia del volumen: {volume['obv_trend']}")
            summary_parts.append("")

        # Patrones de velas
        candlesticks = analysis['candlesticks']
        if candlesticks['detected_patterns']:
            summary_parts.append("🕯️ **Patrones de Velas Detectados:**")
            for pattern in candlesticks['detected_patterns'][:3]:  # Mostrar máximo 3 patrones
                pattern_clean = pattern.replace('_', ' ')
                summary_parts.append(f"• {pattern_clean}")
            summary_parts.append("")

        # Niveles clave
        sr = analysis['support_resistance']
        summary_parts.append("🎯 **Niveles Clave:**")
        summary_parts.append(f"• Resistencia: ${sr['resistance_1']:.2f} (a {sr['distance_to_resistance']:.1f}%)")
        summary_parts.append(f"• Soporte: ${sr['support_1']:.2f} (a {sr['distance_to_support']:.1f}%)")
        summary_parts.append("")

        # Recomendación general
        summary_parts.append("💡 **Evaluación General:**")

        # Lógica para recomendación
        bullish_signals = 0
        bearish_signals = 0

        if trend['primary_trend'] == 'Alcista':
            bullish_signals += 1
        elif trend['primary_trend'] == 'Bajista':
            bearish_signals += 1

        if momentum['rsi_condition'] == 'Sobrevendido':
            bullish_signals += 1
        elif momentum['rsi_condition'] == 'Sobrecomprado':
            bearish_signals += 1

        if any('Bullish' in pattern or 'Hammer' in pattern for pattern in candlesticks['detected_patterns']):
            bullish_signals += 1
        if any('Bearish' in pattern or 'Hanging' in pattern for pattern in candlesticks['detected_patterns']):
            bearish_signals += 1

        if bullish_signals > bearish_signals:
            summary_parts.append("El análisis sugiere un sesgo alcista. Considerar posiciones largas con gestión de riesgo apropiada.")
        elif bearish_signals > bullish_signals:
            summary_parts.append("El análisis sugiere un sesgo bajista. Considerar proteger posiciones largas o evaluar posiciones cortas.")
        else:
            summary_parts.append("El mercado muestra señales mixtas. Recomendado mantener cautela y esperar mayor claridad en las señales.")

        # Advertencia de riesgo
        summary_parts.append("")
        summary_parts.append("⚠️ *Este análisis es solo informativo. Siempre use gestión de riesgo y consulte múltiples fuentes antes de tomar decisiones de trading.*")

        return "\n".join(summary_parts)

    def run_complete_analysis(self):
        """
        Ejecuta el análisis completo para los tres horizontes temporales
        """
        # Calcular indicadores
        self.calculate_advanced_indicators()

        # Detectar patrones
        self.detect_candlestick_patterns()

        # Analizar cada horizonte temporal
        results = {}

        for timeframe in ['short', 'medium', 'long']:
            analysis = self.analyze_timeframe(timeframe)
            summary = self.generate_summary(timeframe)

            results[timeframe] = {
                'analysis': analysis,
                'summary': summary
            }

        self.analysis_results = results
        return results

def create_market_analysis_tab(df, symbol):
    """
    Crea la tab de análisis de mercado para integrar con la aplicación principal
    """
    try:
        # Inicializar analizador
        analyzer = MarketAnalyzer(df)

        # Ejecutar análisis completo
        results = analyzer.run_complete_analysis()

        # Crear contenido de la tab (retorna diccionario con los resultados)
        return {
            'analyzer': analyzer,
            'results': results,
            'symbol': symbol,
            'last_updated': datetime.now()
        }

    except Exception as e:
        return {
            'error': f"Error en el análisis: {str(e)}",
            'symbol': symbol,
            'last_updated': datetime.now()
        }

def create_strategy_tab(df, symbol, metrics, selected_strategy, benchmark_data=None, initial_capital=10000):
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

    # Indicadores según estrategia
    if selected_strategy == 'sma_crossover':
        # Medias móviles
        if 'SMA_15' in df.columns:
            fig_main.add_trace(go.Scatter(x=df.index, y=df['SMA_15'], name='SMA 15',
                                    line=dict(color=colors['primary'], width=2)))
        if 'SMA_60' in df.columns:
            fig_main.add_trace(go.Scatter(x=df.index, y=df['SMA_60'], name='SMA 60',
                                    line=dict(color=colors['accent'], width=2)))

        # Señales de compra y venta
        buy_signals = df.dropna(subset=['Buy_Signal'])
        sell_signals = df.dropna(subset=['Sell_Signal'])
        strategy_title = '🎯 Estrategia SMA Crossover con Velas Japonesas'

    elif selected_strategy == 'advanced_scalping':
        # EMAs para scalping
        if 'EMA_8' in df.columns:
            fig_main.add_trace(go.Scatter(x=df.index, y=df['EMA_8'], name='EMA 8',
                                    line=dict(color=colors['primary'], width=1)))
        if 'EMA_21' in df.columns:
            fig_main.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], name='EMA 21',
                                    line=dict(color=colors['secondary'], width=1)))
        if 'EMA_55' in df.columns:
            fig_main.add_trace(go.Scatter(x=df.index, y=df['EMA_55'], name='EMA 55',
                                    line=dict(color=colors['accent'], width=1)))

        # Señales de scalping
        buy_signals = df.dropna(subset=['Scalp_Buy_Signal'])
        sell_signals = df.dropna(subset=['Scalp_Sell_Signal'])
        strategy_title = '⚡ Estrategia Advanced Scalping con Velas Japonesas'
    
    elif selected_strategy == 'frogames':
        # Indicadores para Frogames
        fig_main.add_trace(go.Scatter(x=df.index, y=df['SMA fast'], name='SMA 30',
                                line=dict(color=colors['primary'], width=2)))
        fig_main.add_trace(go.Scatter(x=df.index, y=df['SMA slow'], name='SMA 60',
                                line=dict(color=colors['accent'], width=2)))
        fig_main.add_trace(go.Scatter(x=df.index, y=df['smooth_support'], name='Support',
                                line=dict(color=colors['secondary'], width=1, dash='dot')))
        fig_main.add_trace(go.Scatter(x=df.index, y=df['smooth_resistance'], name='Resistance',
                                line=dict(color='#FF6B6B', width=1, dash='dot')))

        # Señales de Frogames
        buy_signals = df.dropna(subset=['Frogames_Buy_Signal'])
        sell_signals = df.dropna(subset=['Frogames_Sell_Signal'])
        strategy_title = '🐸 Frogames Strategy'
    
    if selected_strategy == 'sma_crossover':
        buy_col = 'Buy_Signal'
        sell_col = 'Sell_Signal'
    elif selected_strategy == 'advanced_scalping':
        buy_col = 'Scalp_Buy_Signal'
        sell_col = 'Scalp_Sell_Signal'
    elif selected_strategy == 'frogames':
        buy_col = 'Frogames_Buy_Signal'
        sell_col = 'Frogames_Sell_Signal'

    # Get signals safely
    buy_signals = df.dropna(subset=[buy_col]) if buy_col in df.columns else pd.DataFrame()
    sell_signals = df.dropna(subset=[sell_col]) if sell_col in df.columns else pd.DataFrame()

    # Plotear señales
    if not buy_signals.empty:
        fig_main.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals[buy_col],
            mode='markers',
            name='🟢 Comprar',
            marker=dict(
                color='lime',
                size=8 if selected_strategy == 'advanced_scalping' else 12,
                symbol='triangle-up',
                line=dict(width=2, color='white')
            )
        ))

    if not sell_signals.empty:
        fig_main.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals[sell_col],
            mode='markers',
            name='🔴 Vender',
            marker=dict(
                color='red',
                size=8 if selected_strategy == 'advanced_scalping' else 12,
                symbol='triangle-down',
                line=dict(width=2, color='white')
            )
        ))
    
    if selected_strategy == 'sma_crossover' and 'SMA_15' in df.columns and 'SMA_60' in df.columns:
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
        title=f'{strategy_title} - {symbol}',
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

    # Gráfico de Equity Curve
    fig_equity = go.Figure()

    fig_equity.add_trace(go.Scatter(x=df.index, y=df['Portfolio_Value'], name='Estrategia',
                                   line=dict(color=colors['primary'], width=2)))

    if benchmark_data is not None:
        benchmark_aligned = benchmark_data.reindex(df.index, method='ffill')
        benchmark_portfolio = initial_capital * (benchmark_aligned / benchmark_aligned.iloc[0])
        fig_equity.add_trace(go.Scatter(x=df.index, y=benchmark_portfolio, name='Benchmark (S&P 500)',
                                       line=dict(color=colors['accent'], width=2)))

    fig_equity.update_layout(
        title=f'📈 Curva de Capital - {symbol}',
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        xaxis_gridcolor=colors['grid'],
        yaxis_gridcolor=colors['grid'],
        height=300,
        yaxis_title="Valor ($)"
    )

    fig_equity.update_xaxes(gridcolor=colors['grid'])
    fig_equity.update_yaxes(gridcolor=colors['grid'])

    # Estadísticas de señales más detalladas
    total_buy_signals = len(buy_signals)
    total_sell_signals = len(sell_signals)
    total_signals = total_buy_signals + total_sell_signals
    
    # Análisis del estado actual de la estrategia
    if selected_strategy == 'sma_crossover':
        current_sma15 = df['SMA_15'].iloc[-1] if len(df) > 0 and 'SMA_15' in df.columns and not df['SMA_15'].isna().iloc[-1] else 0
        current_sma60 = df['SMA_60'].iloc[-1] if len(df) > 0 and 'SMA_60' in df.columns and not df['SMA_60'].isna().iloc[-1] else 0
        current_position = "ALCISTA" if current_sma15 > current_sma60 else "BAJISTA"
        position_color = colors['secondary'] if current_sma15 > current_sma60 else '#FF6B6B'
    elif selected_strategy == 'advanced_scalping':
        current_sma15 = 0
        current_sma60 = 0
        current_position = "SCALPING ACTIVO"
        position_color = colors['accent']
    elif selected_strategy == 'frogames':
        current_sma15 = df['SMA fast'].iloc[-1] if len(df) > 0 and 'SMA fast' in df.columns and not df['SMA fast'].isna().iloc[-1] else 0
        current_sma60 = df['SMA slow'].iloc[-1] if len(df) > 0 and 'SMA slow' in df.columns and not df['SMA slow'].isna().iloc[-1] else 0
        current_position = "ALCISTA" if current_sma15 > current_sma60 else "BAJISTA"
        position_color = colors['secondary'] if current_sma15 > current_sma60 else '#FF6B6B'
    
    # Calcular días desde última señal
    last_signal_date = None
    if not buy_signals.empty or not sell_signals.empty:
        if selected_strategy == 'sma_crossover':
            signal_cols = ['Buy_Signal', 'Sell_Signal']
        elif selected_strategy == 'advanced_scalping':
            signal_cols = ['Scalp_Buy_Signal', 'Scalp_Sell_Signal']
        elif selected_strategy == 'frogames':
            signal_cols = ['Frogames_Buy_Signal', 'Frogames_Sell_Signal']
        all_signals = pd.concat([buy_signals[signal_cols], sell_signals[signal_cols]])
        last_signal_date = all_signals.index.max()
        days_since_signal = (df.index.max() - last_signal_date).days
    else:
        days_since_signal = "N/A"

    # Definir descripciones de señales
    buy_signal_desc = "Cuando SMA 15 cruza por encima de SMA 60 (tendencia alcista)"
    sell_signal_desc = "Cuando SMA 15 cruza por debajo de SMA 60 (tendencia bajista)"
    if selected_strategy == 'advanced_scalping':
        buy_signal_desc = "Combinación de EMAs, RSI, Estocástico, MACD y patrones de velas en horario de alta liquidez"
        sell_signal_desc = "Stop Loss, Take Profit, o cambio de tendencia detectado"
    elif selected_strategy == 'frogames':
        buy_signal_desc = "Precio rompe resistencia suavizada (close > resistance*1.005), SMA 30 > SMA 60, RSI decreciente"
        sell_signal_desc = "Precio rompe soporte suavizado (close < support*0.995), SMA 30 < SMA 60, RSI creciente"

    # Tooltips informativos
    tooltip_style = {
        'cursor': 'help',
        'textDecoration': 'none',
        'borderBottom': '1px dotted #669FEE'
    }

    return html.Div([
        dcc.Graph(figure=fig_main),
        
        html.Div([
            html.H3(f"🎯 Análisis de Estrategia {'SMA Crossover' if selected_strategy == 'sma_crossover' else 'Advanced Scalping' if selected_strategy == 'advanced_scalping' else 'Frogames'}", style={'color': colors['text']}),
            
            # Fila superior con métricas principales
            html.Div([
                html.Div([
                    html.H4(f"{total_buy_signals}", style={'color': 'lime', 'margin': 0, 'fontSize': '2em'}),
                    html.P("Señales de Compra", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px', 'minWidth': '150px'}),
                
                html.Div([
                    html.H4(f"{total_sell_signals}", style={'color': 'red', 'margin': 0, 'fontSize': '2em'}),
                    html.P("Señales de Venta", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px', 'minWidth': '150px'}),
                
                html.Div([
                    html.H4(f"{total_signals}", style={'color': colors['accent'], 'margin': 0, 'fontSize': '2em'}),
                    html.P("Total Operaciones", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px', 'minWidth': '150px'}),
                
                html.Div([
                    html.H4(f"{current_position}", style={'color': position_color, 'margin': 0, 'fontSize': '1.5em'}),
                    html.P("Tendencia Actual", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px', 'minWidth': '150px'})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap', 'marginBottom': '20px'}),
            
            # Fila inferior con datos técnicos
            html.Div([
                html.Div([
                    html.H4(f"${current_sma15:.2f}", style={'color': colors['primary'], 'margin': 0}),
                    html.P("SMA 15 Actual" if selected_strategy == 'sma_crossover' else "SMA 30 Actual" if selected_strategy == 'frogames' else "SMA 15 Actual", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H4(f"${current_sma60:.2f}", style={'color': colors['accent'], 'margin': 0}),
                    html.P("SMA 60 Actual" if selected_strategy != 'frogames' else "SMA Slow Actual", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H4(f"{((current_sma15 / current_sma60 - 1) * 100):.2f}%", 
                           style={'color': 'lime' if current_sma15 > current_sma60 else 'red', 'margin': 0}),
                    html.P("Divergencia SMAs", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H4(f"{days_since_signal}", style={'color': colors['text'], 'margin': 0}),
                    html.P("Días desde última señal", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}),
        ]),
        
        # Gráfico de Drawdown
        html.Div([
            dcc.Graph(figure=fig_drawdown),
            
            # Métricas de drawdown al lado del gráfico
            html.Div([
                html.Div([
                    html.H4(f"{metrics['max_drawdown']:.1f}%", style={'color': '#FF6B6B', 'margin': 0, 'fontSize': '2em'}),
                    html.P("Max Drawdown", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H4(f"{df['Drawdown'].mean():.1f}%", style={'color': colors['accent'], 'margin': 0, 'fontSize': '1.8em'}),
                    html.P("Drawdown Promedio", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'})
            ], style={'display': 'flex', 'justifyContent': 'center', 'marginTop': '20px'})
        ], style={'marginTop': '30px'}),

        # Gráfico de Equity Curve
        html.Div([
            dcc.Graph(figure=fig_equity),
        ], style={'marginTop': '30px'}),

        # Sección de Métricas Avanzadas de Backtesting
        html.Div([
            html.H3("📊 Métricas Avanzadas de Backtesting", style={'color': colors['text'], 'marginTop': '30px'}),
            
            # Rendimiento
            html.H4("💰 Rendimiento", style={'color': colors['secondary'], 'marginTop': '20px'}),
            html.Div([
                html.Div([
                    html.H4(f"{metrics['total_return']:.1f}%", 
                           style={'color': 'green' if metrics['total_return'] > 0 else 'red', 'margin': 0}),
                    html.P([
                        "Retorno Total ",
                        html.Span("ℹ️", title="Ganancia o pérdida total de la estrategia desde el inicio", style=tooltip_style)
                    ], style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H4(f"{metrics['annual_return']:.1f}%", 
                           style={'color': 'green' if metrics['annual_return'] > 0 else 'red', 'margin': 0}),
                    html.P([
                        "Retorno Anualizado ",
                        html.Span("ℹ️", title="Retorno promedio que la estrategia habría generado si se mantuviera durante un año completo", style=tooltip_style)
                    ], style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}),
            
            # Riesgo
            html.H4("⚠️ Métricas de Riesgo", style={'color': '#FF6B6B', 'marginTop': '20px'}),
            html.Div([
                html.Div([
                    html.H4(f"{metrics['volatility']:.1f}%", style={'color': colors['accent'], 'margin': 0}),
                    html.P([
                        "Volatilidad Anual ",
                        html.Span("ℹ️", title="Medida de cuánto varían los retornos. Mayor volatilidad = mayor riesgo", style=tooltip_style)
                    ], style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H4(f"{metrics['max_drawdown']:.1f}%", style={'color': '#FF6B6B', 'margin': 0}),
                    html.P([
                        "Max Drawdown ",
                        html.Span("ℹ️", title="Mayor caída desde un pico hasta un valle. Indica la peor racha de pérdidas", style=tooltip_style)
                    ], style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H4(f"{metrics['sortino_ratio']:.2f}", 
                           style={'color': 'green' if metrics['sortino_ratio'] > 1 else colors['text'], 'margin': 0}),
                    html.P([
                        "Ratio Sortino ",
                        html.Span("ℹ️", title="Similar al Sharpe pero solo penaliza la volatilidad negativa. >1 es bueno, >2 es excelente", style=tooltip_style)
                    ], style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}),
            
            # Ajustado al riesgo
            html.H4("⚖️ Métricas Ajustadas al Riesgo", style={'color': colors['primary'], 'marginTop': '20px'}),
            html.Div([
                html.Div([
                    html.H4(f"{metrics['sharpe_ratio']:.2f}", 
                           style={'color': 'green' if metrics['sharpe_ratio'] > 1 else colors['text'], 'margin': 0}),
                    html.P([
                        "Ratio Sharpe ",
                        html.Span("ℹ️", title="Retorno excedente por unidad de riesgo. >1 es bueno, >2 es excelente", style=tooltip_style)
                    ], style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H4(f"{metrics['alpha']:.2f}%", 
                           style={'color': 'green' if metrics['alpha'] > 0 else 'red', 'margin': 0}),
                    html.P([
                        "Alpha ",
                        html.Span("ℹ️", title="Retorno excedente vs el mercado (S&P 500). Positivo = supera al mercado", style=tooltip_style)
                    ], style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H4(f"{metrics['beta']:.2f}", 
                           style={'color': colors['secondary'] if abs(metrics['beta']) < 1 else colors['accent'], 'margin': 0}),
                    html.P([
                        "Beta ",
                        html.Span("ℹ️", title="Sensibilidad al mercado. <1 = menos volátil que S&P 500, >1 = más volátil", style=tooltip_style)
                    ], style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H4(f"{metrics['calmar_ratio']:.2f}", 
                           style={'color': 'green' if metrics['calmar_ratio'] > 0.5 else colors['text'], 'margin': 0}),
                    html.P([
                        "Ratio Calmar ",
                        html.Span("ℹ️", title="Retorno anualizado dividido por max drawdown. Mide retorno por unidad de riesgo de caída", style=tooltip_style)
                    ], style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}),

            # Métricas específicas de scalping (solo si es scalping)
            html.Div([
                html.H4("⚡ Métricas de Scalping", style={'color': colors['accent'], 'marginTop': '20px'}),
                html.Div([
                    html.Div([
                        html.H4(f"{metrics.get('avg_trade_duration', 0):.1f} días", style={'color': colors['primary'], 'margin': 0}),
                        html.P("Duración Promedio de Trade", style={'color': colors['text'], 'margin': 0})
                    ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),

                    html.Div([
                        html.H4(f"{metrics.get('win_loss_ratio', 0):.2f}", style={'color': 'green' if metrics.get('win_loss_ratio', 0) > 1 else colors['text'], 'margin': 0}),
                        html.P("Ratio Ganancias/Pérdidas", style={'color': colors['text'], 'margin': 0})
                    ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),

                    html.Div([
                        html.H4(f"{metrics.get('max_consecutive_losses', 0)}", style={'color': '#FF6B6B', 'margin': 0}),
                        html.P("Máx. Pérdidas Consecutivas", style={'color': colors['text'], 'margin': 0})
                    ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),

                    html.Div([
                        html.H4(f"{metrics.get('profit_factor', 0):.2f}", style={'color': 'green' if metrics.get('profit_factor', 0) > 1 else 'red', 'margin': 0}),
                        html.P("Factor de Profit", style={'color': colors['text'], 'margin': 0})
                    ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'})
                ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'})
            ]) if selected_strategy == 'advanced_scalping' else html.Div(),

            # Información adicional de la estrategia
            html.Div([
                html.H4("ℹ️ Información de la Estrategia", style={'color': colors['text'], 'marginTop': '20px'}),
                html.P(f"• Señal de COMPRA: {buy_signal_desc}", style={'color': colors['text'], 'margin': '5px 0'}),
                html.P(f"• Señal de VENTA: {sell_signal_desc}", style={'color': colors['text'], 'margin': '5px 0'}),
                html.P("• Las velas japonesas muestran la acción del precio detallada (apertura, cierre, máximos, mínimos)",
                      style={'color': colors['text'], 'margin': '5px 0'}),
                html.P(f"• Estado actual: La estrategia está en modo {current_position.lower()}",
                      style={'color': position_color, 'margin': '5px 0', 'fontWeight': 'bold'}),
                html.P("• Benchmark: S&P 500 (^GSPC) para cálculos de Alpha y Beta",
                      style={'color': colors['text'], 'margin': '5px 0'}),
                html.P("• Tasa libre de riesgo: 2% anual para cálculos de Sharpe y Sortino",
                      style={'color': colors['text'], 'margin': '5px 0'})
            ], style={'backgroundColor': '#2A2A2A', 'padding': '15px', 'borderRadius': '10px', 'marginTop': '20px'})
        ])
    ])

def create_price_analysis_tab(df, symbol):
    """Crea la tab de análisis de precio con velas japonesas"""
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=['Precio (Velas Japonesas) y Medias Móviles', 'Volumen'],
                        row_heights=[0.7, 0.3])
    
    # Velas japonesas
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Precio',
        increasing_line_color=colors['secondary'],  # Verde para velas alcistas
        decreasing_line_color='#FF6B6B',  # Rojo para velas bajistas
        increasing_fillcolor=colors['secondary'],
        decreasing_fillcolor='#FF6B6B'
    ), row=1, col=1)
    
    # Medias móviles sobre las velas
    if 'SMA_15' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_15'], name='SMA 15',
                                line=dict(color=colors['primary'], width=2)), row=1, col=1)
    if 'SMA_60' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_60'], name='SMA 60',
                                line=dict(color=colors['accent'], width=2)), row=1, col=1)
    
    # EMA adicional
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20', 
                            line=dict(color='#9988DD', width=1, dash='dot')), row=1, col=1)
    
    # Volumen con colores según el movimiento del precio
    colors_volume = []
    for i in range(len(df)):
        if df['Close'].iloc[i] >= df['Open'].iloc[i]:
            colors_volume.append(colors['secondary'])  # Verde si cierre >= apertura
        else:
            colors_volume.append('#FF6B6B')  # Rojo si cierre < apertura
    
    if 'Volume' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volumen', 
                            marker_color=colors_volume, opacity=0.7), row=2, col=1)
        
        # Media móvil del volumen
        fig.add_trace(go.Scatter(x=df.index, y=df['Volume_SMA'], name='Volumen SMA', 
                                line=dict(color=colors['text'], width=1)), row=2, col=1)
    
    fig.update_layout(
        title=f'Análisis de Precio con Velas Japonesas - {symbol}',
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        xaxis_gridcolor=colors['grid'],
        yaxis_gridcolor=colors['grid'],
        height=700,
        xaxis_rangeslider_visible=False  # Ocultar el rango slider para mejor vista
    )
    
    # Configurar ejes
    fig.update_xaxes(gridcolor=colors['grid'])
    fig.update_yaxes(gridcolor=colors['grid'])
    
    return html.Div([
        dcc.Graph(figure=fig),
        html.Div([
            html.H3("📊 Estadísticas de Velas", style={'color': colors['text']}),
            html.Div([
                html.Div([
                    html.H4(f"${df['Close'].iloc[-1]:.2f}", style={'color': colors['primary'], 'margin': 0}),
                    html.P("Precio Cierre", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '5px'}),
                
                html.Div([
                    html.H4(f"${df['High'].iloc[-1]:.2f}", style={'color': colors['secondary'], 'margin': 0}),
                    html.P("Máximo Día", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '5px'}),
                
                html.Div([
                    html.H4(f"${df['Low'].iloc[-1]:.2f}", style={'color': '#FF6B6B', 'margin': 0}),
                    html.P("Mínimo Día", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '5px'}),
                
                html.Div([
                    html.H4(f"${df['Open'].iloc[-1]:.2f}", style={'color': colors['accent'], 'margin': 0}),
                    html.P("Precio Apertura", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '5px'}),
                
                html.Div([
                    html.H4(f"{((df['Close'].iloc[-1] / df['Open'].iloc[-1] - 1) * 100):.2f}%", 
                           style={'color': 'green' if df['Close'].iloc[-1] >= df['Open'].iloc[-1] else 'red', 'margin': 0}),
                    html.P("Cambio Diario", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '5px'}),
                
                html.Div([
                    html.H4(f"{((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100):.1f}%", 
                           style={'color': colors['secondary'], 'margin': 0}),
                    html.P("Cambio Total", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '5px'})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'})
        ], style={'marginTop': 20})
    ])

def create_indicators_tab(df, symbol):
    """Crea la tab de indicadores técnicos con velas japonesas"""
    fig = make_subplots(rows=4, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=['Precio con Bollinger Bands', 'RSI', 'MACD', 'Volumen'],
                        row_heights=[0.4, 0.2, 0.25, 0.15])
    
    # Velas japonesas con Bollinger Bands
    fig.add_trace(go.Candlestick(
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
    ), row=1, col=1)
    
    # Bollinger Bands
    if 'BB_upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Superior',
                                line=dict(color=colors['accent'], width=1)), row=1, col=1)
    if 'BB_middle' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], name='BB Media',
                                line=dict(color=colors['text'], width=1, dash='dash')), row=1, col=1)
    if 'BB_lower' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Inferior',
                                line=dict(color=colors['accent'], width=1),
                                fill='tonexty', fillcolor='rgba(238, 204, 85, 0.1)'), row=1, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                                line=dict(color=colors['secondary'], width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#FF6B6B", row=2, col=1,
                      annotation_text="Sobrecompra (70)")
        fig.add_hline(y=30, line_dash="dash", line_color=colors['secondary'], row=2, col=1,
                      annotation_text="Sobreventa (30)")
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(105, 159, 238, 0.1)",
                      layer="below", line_width=0, row=2, col=1)

    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                                line=dict(color=colors['primary'], width=2)), row=3, col=1)
    if 'MACD_signal' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Señal',
                                line=dict(color=colors['accent'], width=2)), row=3, col=1)

    # Histograma MACD con colores
    if 'MACD_hist' in df.columns:
        macd_colors = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='Histograma',
                            marker_color=macd_colors, opacity=0.6), row=3, col=1)
    
    # Volumen
    if 'Volume' in df.columns:
        volume_colors = []
        for i in range(len(df)):
            if df['Close'].iloc[i] >= df['Open'].iloc[i]:
                volume_colors.append(colors['secondary'])
            else:
                volume_colors.append('#FF6B6B')

        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volumen',
                            marker_color=volume_colors, opacity=0.7), row=4, col=1)
    
    fig.update_layout(
        title=f'Análisis Técnico Completo - {symbol}',
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        height=900,
        xaxis_rangeslider_visible=False
    )
    
    # Configurar ejes
    fig.update_xaxes(gridcolor=colors['grid'])
    fig.update_yaxes(gridcolor=colors['grid'])
    
    # Añadir información adicional
    current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not df['RSI'].isna().iloc[-1] else 0
    rsi_status = "Sobrecompra" if current_rsi > 70 else "Sobreventa" if current_rsi < 30 else "Neutral"
    rsi_color = "#FF6B6B" if current_rsi > 70 else colors['secondary'] if current_rsi < 30 else colors['accent']

    macd_value = df['MACD'].iloc[-1] if 'MACD' in df.columns and not df['MACD'].isna().iloc[-1] else 0
    bb_upper = df['BB_upper'].iloc[-1] if 'BB_upper' in df.columns and not df['BB_upper'].isna().iloc[-1] else 0
    bb_lower = df['BB_lower'].iloc[-1] if 'BB_lower' in df.columns and not df['BB_lower'].isna().iloc[-1] else 0

    info_panel = html.Div([
        html.H3("🔍 Análisis de Indicadores", style={'color': colors['text']}),
        html.Div([
            html.Div([
                html.H4(f"{current_rsi:.1f}", style={'color': rsi_color, 'margin': 0}),
                html.P(f"RSI - {rsi_status}", style={'color': colors['text'], 'margin': 0})
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),

            html.Div([
                html.H4(f"{macd_value:.4f}", style={'color': colors['primary'], 'margin': 0}),
                html.P("MACD Actual", style={'color': colors['text'], 'margin': 0})
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),

            html.Div([
                html.H4(f"${bb_upper:.2f}", style={'color': colors['accent'], 'margin': 0}),
                html.P("Banda Superior", style={'color': colors['text'], 'margin': 0})
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),

            html.Div([
                html.H4(f"${bb_lower:.2f}", style={'color': colors['accent'], 'margin': 0}),
                html.P("Banda Inferior", style={'color': colors['text'], 'margin': 0})
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'})
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'})
    ], style={'marginTop': 20})
    
    return html.Div([
        dcc.Graph(figure=fig),
        info_panel
    ])

def create_performance_tab(df, symbol, capital, benchmark_data=None):
    """Crea la tab de rendimiento y backtesting"""
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=['Valor del Portfolio vs Buy & Hold', 'Retornos Diarios'])
    
    # Portfolio vs Buy & Hold
    buy_hold_value = capital * (df['Close'] / df['Close'].iloc[0])

    fig.add_trace(go.Scatter(x=df.index, y=df['Portfolio_Value'], name='Estrategia',
                            line=dict(color=colors['primary'], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=buy_hold_value, name='Buy & Hold',
                            line=dict(color=colors['accent'], width=2)), row=1, col=1)

    if benchmark_data is not None:
        benchmark_aligned = benchmark_data.reindex(df.index, method='ffill')
        benchmark_portfolio = capital * (benchmark_aligned / benchmark_aligned.iloc[0])
        fig.add_trace(go.Scatter(x=df.index, y=benchmark_portfolio, name='Benchmark (S&P 500)',
                                line=dict(color=colors['secondary'], width=2)), row=1, col=1)
    
    # Retornos diarios
    fig.add_trace(go.Scatter(x=df.index, y=df['Strategy_Returns'] * 100, name='Retornos Estrategia (%)', 
                            line=dict(color=colors['secondary'], width=1)), row=2, col=1)
    
    # Retorno acumulado
    acumulado = df['Strategy_Returns'].cumsum() 
    fig.add_trace(go.Scatter(x=df.index, y=acumulado * 100, name='Retorno Acumulado', 
                            line=dict(color=colors['secondary'], width=1)), row=2, col=1)
    
    fig.update_layout(
        title=f'Análisis de Rendimiento - {symbol}',
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        height=700
    )
    
    # Calcular métricas de rendimiento
    final_value = df['Portfolio_Value'].iloc[-1]
    total_return = (final_value / capital - 1) * 100
    buy_hold_return = (buy_hold_value.iloc[-1] / capital - 1) * 100
    
    annual_return = ((final_value / capital) ** (365.25 / len(df)) - 1) * 100
    volatility = df['Strategy_Returns'].std() * np.sqrt(252) * 100
    sharpe_ratio = (annual_return / volatility) if volatility != 0 else 0
    
    max_drawdown = ((df['Portfolio_Value'] / df['Portfolio_Value'].cummax()) - 1).min() * 100
    
    return html.Div([
        dcc.Graph(figure=fig),
        html.Div([
            html.H3("📊 Métricas de Rendimiento", style={'color': colors['text']}),
            html.Div([
                html.Div([
                    html.H4(f"${final_value:,.0f}", style={'color': colors['primary'], 'margin': 0}),
                    html.P("Valor Final", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H4(f"{total_return:.1f}%", 
                           style={'color': 'green' if total_return > 0 else 'red', 'margin': 0}),
                    html.P("Retorno Total", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H4(f"{buy_hold_return:.1f}%", 
                           style={'color': 'green' if buy_hold_return > 0 else 'red', 'margin': 0}),
                    html.P("Buy & Hold", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H4(f"{sharpe_ratio:.2f}", style={'color': colors['secondary'], 'margin': 0}),
                    html.P("Ratio Sharpe", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H4(f"{max_drawdown:.1f}%", style={'color': '#FF6B6B', 'margin': 0}),
                    html.P("Max Drawdown", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'}),
                
                html.Div([
                    html.H4(f"{volatility:.1f}%", style={'color': colors['accent'], 'margin': 0}),
                    html.P("Volatilidad Anual", style={'color': colors['text'], 'margin': 0})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#2A2A2A', 'borderRadius': '10px', 'margin': '10px'})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'})
        ])
    ])

# Layout de la aplicación
app.layout = html.Div([
    html.Div([
        html.H1("🚀 Trading Dashboard Algorítmico", 
                style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': 30}),
        
        # Controles superiores
        html.Div([
            html.Div([
                html.Label("Símbolo:", style={'color': colors['text']}),
                dcc.Dropdown(
                    id='symbol-dropdown',
                    options=[
                        {'label': 'Bitcoin (BTC-USD)', 'value': 'BTC-USD'},
                        {'label': 'Ethereum (ETH-USD)', 'value': 'ETH-USD'},
                        {'label': 'Apple (AAPL)', 'value': 'AAPL'},
                        {'label': 'Tesla (TSLA)', 'value': 'TSLA'},
                        {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
                    ],
                    value='BTC-USD',
                    style={'backgroundColor': colors['background'], 'color': colors['text']}
                )
            ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.Label("Período:", style={'color': colors['text']}),
                dcc.Dropdown(
                    id='period-dropdown',
                    options=[
                        {'label': '1 Mes', 'value': '1mo'},
                        {'label': '3 Meses', 'value': '3mo'},
                        {'label': '6 Meses', 'value': '6mo'},
                        {'label': '1 Año', 'value': '1y'},
                        {'label': '2 Años', 'value': '2y'},
                        {'label': 'Máximo', 'value': 'max'}
                    ],
                    value='1y',
                    style={'backgroundColor': colors['background'], 'color': colors['text']}
                )
            ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                html.Label("Capital Inicial ($):", style={'color': colors['text']}),
                dcc.Input(
                    id='capital-input',
                    type='number',
                    value=10000,
                    min=1000,
                    step=1000,
                    style={'backgroundColor': colors['background'], 'color': colors['text'],
                           'border': f'1px solid {colors["grid"]}', 'width': '100%'}
                )
            ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}),

            html.Div([
                html.Label("Estrategia:", style={'color': colors['text']}),
                dcc.Dropdown(
                    id='strategy-dropdown',
                    options=[
                        {'label': 'SMA Crossover', 'value': 'sma_crossover'},
                        {'label': 'Advanced Scalping', 'value': 'advanced_scalping'},
                        {'label': 'Frogames Strategy', 'value': 'frogames'}
                    ],
                    value='sma_crossover',
                    style={'backgroundColor': colors['background'], 'color': colors['text']}
                )
            ], style={'width': '18%', 'display': 'inline-block', 'marginRight': '2%'}),

            html.Div([
                html.Button('🔄 Actualizar Datos', id='update-button', n_clicks=0,
                           style={'backgroundColor': colors['primary'], 'color': 'white',
                                  'border': 'none', 'padding': '10px 20px', 'borderRadius': '5px',
                                  'cursor': 'pointer', 'fontSize': '14px', 'marginTop': '25px'})
            ], style={'width': '20%', 'display': 'inline-block'})
        ], style={'marginBottom': 30}),

        # Controles específicos de scalping (ocultos por defecto)
        html.Div(id='scalping-controls', children=[
            html.H4("⚡ Parámetros de Scalping", style={'color': colors['text'], 'marginTop': '20px'}),
            html.Div([
                html.Div([
                    html.Label("Stop Loss ATR Multiplier:", style={'color': colors['text']}),
                    dcc.Input(
                        id='stop-loss-atr-input',
                        type='number',
                        value=1.5,
                        min=0.1,
                        step=0.1,
                        style={'backgroundColor': colors['background'], 'color': colors['text'],
                               'border': f'1px solid {colors["grid"]}', 'width': '100%'}
                    )
                ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '1%'}),

                html.Div([
                    html.Label("Take Profit Ratio:", style={'color': colors['text']}),
                    dcc.Input(
                        id='take-profit-ratio-input',
                        type='number',
                        value=2.0,
                        min=0.1,
                        step=0.1,
                        style={'backgroundColor': colors['background'], 'color': colors['text'],
                               'border': f'1px solid {colors["grid"]}', 'width': '100%'}
                    )
                ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '1%'}),

                html.Div([
                    html.Label("Position Size %:", style={'color': colors['text']}),
                    dcc.Input(
                        id='position-size-pct-input',
                        type='number',
                        value=2,
                        min=0.1,
                        step=0.1,
                        style={'backgroundColor': colors['background'], 'color': colors['text'],
                               'border': f'1px solid {colors["grid"]}', 'width': '100%'}
                    )
                ], style={'width': '23%', 'display': 'inline-block', 'marginRight': '1%'}),

                html.Div([
                    html.Label("Max Concurrent Positions:", style={'color': colors['text']}),
                    dcc.Input(
                        id='max-positions-input',
                        type='number',
                        value=3,
                        min=1,
                        step=1,
                        style={'backgroundColor': colors['background'], 'color': colors['text'],
                               'border': f'1px solid {colors["grid"]}', 'width': '100%'}
                    )
                ], style={'width': '23%', 'display': 'inline-block'})
            ], style={'marginBottom': '20px'})
        ], style={'display': 'none'}),  # Inicialmente oculto

        # Tabs para diferentes vistas
        dcc.Tabs(id='main-tabs', value='price-analysis', children=[
            dcc.Tab(label='📈 Análisis de Precio', value='price-analysis'),
            dcc.Tab(label='📊 Indicadores Técnicos', value='indicators'),
            dcc.Tab(label='🎯 Estrategia & Backtesting', value='strategy'),
            dcc.Tab(label='📈 Rendimiento', value='performance'),
            dcc.Tab(label='🕐 Análisis del Tiempo', value='market-analysis')
        ], style={'backgroundColor': colors['background']}),
        
        # Contenido principal
        html.Div(id='tab-content', style={'marginTop': 20})
        
    ], style={'padding': '20px', 'backgroundColor': colors['background'], 'minHeight': '100vh'})
])

# Callback para mostrar/ocultar controles de scalping
@app.callback(
    Output('scalping-controls', 'style'),
    [Input('strategy-dropdown', 'value')]
)
def toggle_scalping_controls(selected_strategy):
    if selected_strategy == 'advanced_scalping':
        return {'display': 'block'}
    else:
        return {'display': 'none'}

# Callback para actualizar el contenido de las tabs
@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'value'),
     Input('symbol-dropdown', 'value'),
     Input('period-dropdown', 'value'),
     Input('capital-input', 'value'),
     Input('strategy-dropdown', 'value'),
     Input('stop-loss-atr-input', 'value'),
     Input('take-profit-ratio-input', 'value'),
     Input('position-size-pct-input', 'value'),
     Input('max-positions-input', 'value'),
     Input('update-button', 'n_clicks')],
    prevent_initial_call=False
)
def update_tab_content(active_tab, symbol, period, capital, selected_strategy,
                      stop_loss_atr, take_profit_ratio, position_size_pct, max_positions, n_clicks):
    try:
        # Validar inputs
        if not symbol or not period or not capital:
            return html.Div("Por favor, complete todos los campos",
                          style={'color': colors['text'], 'textAlign': 'center', 'padding': '50px'})

        # Obtener y procesar datos
        df = get_crypto_data(symbol, period)
        if df.empty:
            return html.Div("Error cargando datos. Verifique el símbolo y conexión a internet.",
                          style={'color': colors['text'], 'textAlign': 'center', 'padding': '50px'})

        df = calculate_indicators(df)

        # Aplicar estrategia seleccionada
        if selected_strategy == 'sma_crossover':
            df = sma_crossover_strategy(df)
        elif selected_strategy == 'advanced_scalping':
            risk_config = {
                'stop_loss_atr': stop_loss_atr or 1.5,
                'take_profit_ratio': take_profit_ratio or 2.0,
                'position_size': (position_size_pct or 2) / 100,  # Convertir a decimal
                'max_positions': max_positions or 3
            }
            df = advanced_scalping_strategy(df, risk_config)
            df['Position'] = df['Scalp_Position']  # Ensure Position column for backtesting
        elif selected_strategy == 'frogames':
            df = frogames_strategy(df)

        # Obtener datos del benchmark para métricas avanzadas
        benchmark_data = get_benchmark_data("^GSPC", period)
        df, metrics = backtest_strategy(df, benchmark_data, capital)
        
        if active_tab == 'price-analysis':
            return create_price_analysis_tab(df, symbol)
        elif active_tab == 'indicators':
            return create_indicators_tab(df, symbol)
        elif active_tab == 'strategy':
            return create_strategy_tab(df, symbol, metrics, selected_strategy, benchmark_data, capital)
        elif active_tab == 'performance':
            return create_performance_tab(df, symbol, capital, benchmark_data)
        elif active_tab == 'market-analysis':
            analysis_result = create_market_analysis_tab(df, symbol)
            if 'error' in analysis_result:
                return html.Div(f"Error: {analysis_result['error']}",
                              style={'color': 'red', 'textAlign': 'center', 'padding': '50px'})
            else:
                # Create a proper HTML layout for the market analysis
                return html.Div([
                    html.H3(f"🕐 Análisis de Mercado Multi-Temporal - {symbol}",
                           style={'color': colors['text'], 'textAlign': 'center'}),
                    html.Div([
                        html.Div([
                            html.H4("📊 Corto Plazo (1-5 días)", style={'color': colors['primary']}),
                            html.Pre(analysis_result['results']['short']['summary'],
                                   style={'backgroundColor': '#2A2A2A', 'color': colors['text'],
                                          'padding': '15px', 'borderRadius': '10px', 'whiteSpace': 'pre-wrap'})
                        ], style={'margin': '10px'}),
                        html.Div([
                            html.H4("📊 Mediano Plazo (5-20 días)", style={'color': colors['secondary']}),
                            html.Pre(analysis_result['results']['medium']['summary'],
                                   style={'backgroundColor': '#2A2A2A', 'color': colors['text'],
                                          'padding': '15px', 'borderRadius': '10px', 'whiteSpace': 'pre-wrap'})
                        ], style={'margin': '10px'}),
                        html.Div([
                            html.H4("📊 Largo Plazo (20+ días)", style={'color': colors['accent']}),
                            html.Pre(analysis_result['results']['long']['summary'],
                                   style={'backgroundColor': '#2A2A2A', 'color': colors['text'],
                                          'padding': '15px', 'borderRadius': '10px', 'whiteSpace': 'pre-wrap'})
                        ], style={'margin': '10px'})
                    ])
                ])
        else:
            return html.Div("Pestaña no encontrada",
                          style={'color': colors['text'], 'textAlign': 'center', 'padding': '50px'})
            
    except Exception as e:
        return html.Div(f"Error: {str(e)}", 
                      style={'color': 'red', 'textAlign': 'center', 'padding': '50px'})

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)