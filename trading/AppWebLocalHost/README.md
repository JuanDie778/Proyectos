# Panel de Trading - Plataforma de Análisis Algorítmico

## 📊 Resumen del Proyecto

Este proyecto implementa un panel de trading algorítmico integral diseñado para análisis cuantitativo y desarrollo de estrategias. Como Científico de Datos especializado en mercados financieros, he desarrollado una plataforma sofisticada que combina análisis técnico avanzado, evaluación de mercado multi-temporal y metodologías de backtesting cuantitativo para apoyar decisiones de trading basadas en datos.

La plataforma integra reconocimiento de patrones inspirado en aprendizaje automático, análisis estadístico y marcos de gestión de riesgos para proporcionar insights accionables tanto para estrategias de scalping a corto plazo como de inversión a largo plazo.

## 🎯 Características Principales

### Análisis Técnico Avanzado
- **Análisis Multi-Temporal**: Evaluación de mercado a corto plazo (1-5 días), medio plazo (5-20 días) y largo plazo (20+ días)
- **Suite Completa de Indicadores**: RSI, MACD, Bandas de Bollinger, Oscilador Estocástico, Nube Ichimoku, ATR e indicadores personalizados
- **Reconocimiento de Patrones de Velas**: Detección automatizada de más de 8 patrones de velas japonesas, incluyendo Hammer, Engulfing, Harami y formaciones Marubozu

### Marco de Estrategias Cuantitativas
- **Estrategia de Cruce SMA**: Seguimiento de tendencias basado en momentum con cruces de medias móviles
- **Estrategia Avanzada de Scalping**: Enfoque de trading de alta frecuencia utilizando EMAs, RSI, Estocástico y convergencia MACD
- **Estrategia Frogames**: Sistema de ruptura de soporte/resistencia con confirmación de momentum RSI

### Gestión de Riesgos y Backtesting
- **Cálculo Avanzado de Métricas**: Ratio Sharpe, Ratio Sortino, Drawdown Máximo, análisis Alpha/Beta
- **Comparación con Benchmarks**: Evaluación de rendimiento contra S&P 500 (^GSPC)
- **Dimensionamiento de Posiciones**: Cálculos de stop-loss y take-profit basados en ATR
- **Análisis de Drawdown**: Monitoreo de riesgo de portafolio en tiempo real

### Metodología de Ciencia de Datos

#### Ingeniería de Características
- **Indicadores Técnicos**: Más de 15 indicadores calculados incluyendo métricas de momentum, volatilidad y volumen
- **Reconocimiento de Patrones**: Algoritmos de detección de patrones de velas inspirados en aprendizaje automático
- **Análisis de Series Temporales**: Estadísticas móviles y cálculos de fuerza de tendencia

#### Análisis Estadístico
- **Métricas de Rendimiento**: Cálculos completos de retornos ajustados por riesgo
- **Análisis de Correlación**: Rendimiento de estrategias vs. benchmarks de mercado
- **Modelado de Volatilidad**: Análisis ATR y ancho de Bandas de Bollinger para detección de regímenes de mercado

#### Integración de Aprendizaje Automático
- **Puntuación de Señales**: Sistema de puntuación multi-factor para generación de señales de trading
- **Clasificación de Patrones**: Identificación automatizada de condiciones de mercado alcistas/bajistas
- **Parámetros Adaptativos**: Ajustes dinámicos de umbrales basados en volatilidad de mercado

## 🏗️ Arquitectura

### Componentes Principales

#### 1. Capa de Adquisición de Datos
```python
# Obtención de datos en tiempo real desde Yahoo Finance
def get_crypto_data(symbol="BTC-USD", period="1y"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    return df
```

#### 2. Motor de Análisis Técnico
- **Clase MarketAnalyzer**: Análisis multi-temporal con evaluación de tendencia, momentum y volatilidad
- **Cálculo de Indicadores**: Integración TA-Lib para indicadores técnicos de grado profesional
- **Detección de Patrones**: Algoritmos basados en reglas para identificación de patrones de velas

#### 3. Implementación de Estrategias
- **Diseño Modular de Estrategias**: Clases separadas para diferentes enfoques de trading
- **Integración de Gestión de Riesgos**: Dimensionamiento de posiciones y colocación de stop-loss basados en ATR
- **Marco de Backtesting**: Evaluación de rendimiento histórico con métricas avanzadas

#### 4. Panel de Visualización
- **Framework Dash**: Interfaz web interactiva para análisis en tiempo real
- **Integración Plotly**: Gráficos avanzados con velas, indicadores y métricas de rendimiento
- **Interfaz Multi-Pestaña**: Vistas organizadas para análisis de precios, indicadores, estrategias y rendimiento

### Arquitectura de Flujo de Datos
```
Datos Crudos de Mercado → Indicadores Técnicos → Señales de Estrategia → Gestión de Riesgos → Métricas de Rendimiento → Visualización
```

## 📈 Métodos Cuantitativos

### Cálculos de Indicadores Técnicos
- **Medias Móviles**: SMA, EMA con múltiples períodos para identificación de tendencias
- **Indicadores de Momentum**: RSI, Estocástico, MACD para condiciones de sobrecompra/sobreventa
- **Medidas de Volatilidad**: Bandas de Bollinger, ATR para evaluación de riesgos
- **Análisis de Volumen**: OBV, SMA de Volumen para análisis de participación de mercado

### Evaluación Estadística de Rendimiento
- **Métricas de Retorno**: Retorno total, retorno anualizado, retornos ajustados por riesgo
- **Métricas de Riesgo**: Volatilidad, drawdown máximo, Valor en Riesgo (VaR)
- **Análisis de Benchmarks**: Cálculos Alpha/Beta contra índices de mercado
- **Análisis de Trades**: Ratio ganancia/pérdida, factor de beneficio, duración promedio de trades

### Enfoques de Aprendizaje Automático
- **Reconocimiento de Patrones**: Clasificación basada en reglas de patrones de mercado
- **Filtrado de Señales**: Puntuación multi-condición para calidad de señales de trading
- **Algoritmos Adaptativos**: Detección de regímenes de mercado para ajuste de estrategias

## 🚀 Instalación

### Prerrequisitos
- Python 3.8+
- Gestor de paquetes pip

### Dependencias
```bash
pip install -r requirements.txt
```

Bibliotecas clave incluyen:
- **dash**: Framework web para paneles interactivos
- **plotly**: Gráficos y visualización avanzados
- **yfinance**: Adquisición de datos financieros
- **ta**: Indicadores de análisis técnico
- **pandas/numpy**: Manipulación de datos y computación numérica
- **scikit-learn**: Utilidades de aprendizaje automático

### Configuración
1. Clonar el repositorio
2. Instalar dependencias: `pip install -r requirements.txt`
3. Ejecutar la aplicación: `python principal.py`
4. Acceder al panel en `http://127.0.0.1:8050`

## 📊 Uso

### Interfaz del Panel
La plataforma proporciona cinco pestañas analíticas principales:

#### 1. Pestaña de Análisis de Precios
- Gráficos de velas con patrones japoneses
- Medias móviles y líneas de tendencia
- Análisis de volumen con barras codificadas por color

#### 2. Pestaña de Indicadores Técnicos
- Bandas de Bollinger con análisis de posición
- RSI con niveles de sobrecompra/sobreventa
- MACD con línea de señal e histograma
- Indicadores de volumen

#### 3. Pestaña de Estrategias y Backtesting
- Señales de estrategia en tiempo real
- Panel de métricas de rendimiento
- Análisis de drawdown
- Visualización de curva de equity

#### 4. Pestaña de Rendimiento
- Valor de portafolio vs. comparación buy-and-hold
- Análisis de rendimiento de benchmarks
- Métricas de retorno ajustadas por riesgo

#### 5. Pestaña de Análisis de Mercado
- Análisis de tendencias multi-temporal
- Evaluación de momentum y volatilidad
- Identificación de niveles de soporte/resistencia
- Resúmenes de mercado automatizados

### Configuración de Estrategias
- **Asignación de Capital**: Capital inicial ajustable (predeterminado: $10,000)
- **Parámetros de Riesgo**: Multiplicadores ATR para stop-loss/take-profit
- **Límites de Posición**: Posiciones concurrentes máximas para scalping
- **Filtros de Tiempo**: Restricciones de horas de alta liquidez para estrategias de scalping

## 🔬 Insights de Ciencia de Datos

### Detección de Regímenes de Mercado
La plataforma implementa detección sofisticada de regímenes de mercado utilizando:
- **Análisis de Volatilidad**: Ancho de Bandas de Bollinger para evaluación de condiciones de mercado
- **Fuerza de Tendencia**: Análisis de separación de medias móviles
- **Cambios de Momentum**: Detección de divergencias RSI y MACD

### Evaluación Cuantitativa de Estrategias
- **Análisis de Ratio Sharpe**: Optimización de retornos ajustados por riesgo
- **Gestión de Drawdown**: Límites de pérdida máxima y análisis de recuperación
- **Comparación con Benchmarks**: Medición de sobre-rendimiento contra índices de mercado

### Algoritmos de Reconocimiento de Patrones
- **Clasificación de Velas**: Análisis geométrico de patrones de acción de precio
- **Detección de Soporte/Resistencia**: Cálculos de puntos pivote y validación
- **Identificación de Rupturas**: Confirmación de volumen y análisis de momentum

## 🤝 Contribuyendo

### Directrices de Desarrollo
1. **Calidad de Código**: Seguir estándares PEP 8 e incluir docstrings completos
2. **Pruebas**: Implementar pruebas unitarias para nuevos indicadores y estrategias
3. **Documentación**: Actualizar README y documentación en línea para nuevas características
4. **Rendimiento**: Optimizar cálculos para capacidades de análisis en tiempo real

### Áreas de Mejora de Características
- **Integración de Aprendizaje Automático**: Modelos LSTM para predicción de precios
- **Fuentes de Datos Alternativas**: Análisis de sentimiento de noticias y redes sociales
- **Optimización de Portafolio**: Implementación de Teoría Moderna de Portafolio
- **Trading de Alta Frecuencia**: Capacidades de procesamiento de datos tick-level

### Direcciones de Investigación
- **Aprendizaje Profundo**: Reconocimiento de patrones basado en redes neuronales
- **Aprendizaje por Refuerzo**: Optimización automatizada de estrategias
- **Procesamiento de Lenguaje Natural**: Análisis de sentimiento de noticias financieras
- **Inversiones Alternativas**: Análisis de criptomonedas y derivados

## 📄 Licencia

Este proyecto se desarrolla para fines educativos e investigativos en finanzas cuantitativas y trading algorítmico. Por favor, asegúrese de cumplir con regulaciones locales e implementar prácticas apropiadas de gestión de riesgos antes de desplegar en entornos de trading en vivo.

## ⚠️ Descargo de Responsabilidad

Esta plataforma está diseñada para fines educativos e investigativos. Todo trading implica riesgo, y el rendimiento pasado no garantiza resultados futuros. Siempre realice backtesting exhaustivo y testing forward antes de implementar cualquier estrategia en mercados en vivo. Los desarrolladores no son responsables de pérdidas financieras incurridas por el uso de este software.

---

**Desarrollado con ❤️ para la comunidad de finanzas cuantitativas**