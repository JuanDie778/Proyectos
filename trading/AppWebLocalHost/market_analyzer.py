import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

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

# Función auxiliar para integración con la aplicación principal
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