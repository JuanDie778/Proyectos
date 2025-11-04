# Estas funciones van insertadas en el archivo principal.py donde dice "# [Código mantenido por brevedad - usar el código original]"

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
        fig_main.add_trace(go.Scatter(x=df.index, y=df['SMA_15'], name='SMA 15',
                                line=dict(color=colors['primary'], width=2)))
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
    
    if selected_strategy == 'sma_crossover':
        buy_col = 'Buy_Signal'
        sell_col = 'Sell_Signal'
    elif selected_strategy == 'advanced_scalping':
        buy_col = 'Scalp_Buy_Signal'
        sell_col = 'Scalp_Sell_Signal'
    
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
    
    return html.Div([
        dcc.Graph(figure=fig_main),
        html.Div([
            dcc.Graph(figure=fig_drawdown),
        ], style={'marginTop': '30px'}),
        html.Div([
            dcc.Graph(figure=fig_equity),
        ], style={'marginTop': '30px'}),
        # Agregar métricas y análisis adicional aquí
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
        increasing_line_color=colors['secondary'],
        decreasing_line_color='#FF6B6B',
        increasing_fillcolor=colors['secondary'],
        decreasing_fillcolor='#FF6B6B'
    ), row=1, col=1)
    
    # Medias móviles sobre las velas
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_15'], name='SMA 15', 
                            line=dict(color=colors['primary'], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_60'], name='SMA 60', 
                            line=dict(color=colors['accent'], width=2)), row=1, col=1)
    
    # EMA adicional
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20', 
                            line=dict(color='#9988DD', width=1, dash='dot')), row=1, col=1)
    
    # Volumen con colores según el movimiento del precio
    colors_volume = []
    for i in range(len(df)):
        if df['Close'].iloc[i] >= df['Open'].iloc[i]:
            colors_volume.append(colors['secondary'])
        else:
            colors_volume.append('#FF6B6B')
    
    if 'Volume' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volumen', 
                            marker_color=colors_volume, opacity=0.7), row=2, col=1)
        
        # Media móvil del volumen
        if 'Volume_SMA' in df.columns:
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
        xaxis_rangeslider_visible=False
    )
    
    fig.update_xaxes(gridcolor=colors['grid'])
    fig.update_yaxes(gridcolor=colors['grid'])
    
    return html.Div([
        dcc.Graph(figure=fig),
        # Agregar estadísticas adicionales aquí
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
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], name='BB Media', 
                                line=dict(color=colors['text'], width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Inferior', 
                                line=dict(color=colors['accent'], width=1), 
                                fill='tonexty', fillcolor='rgba(238, 204, 85, 0.1)'), row=1, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', 
                                line=dict(color=colors['secondary'], width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#FF6B6B", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=colors['secondary'], row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', 
                                line=dict(color=colors['primary'], width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Señal', 
                                line=dict(color=colors['accent'], width=2)), row=3, col=1)
        
        # Histograma MACD con colores
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
    
    fig.update_xaxes(gridcolor=colors['grid'])
    fig.update_yaxes(gridcolor=colors['grid'])
    
    return html.Div([
        dcc.Graph(figure=fig),
        # Agregar información adicional aquí
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
    
    fig.update_layout(
        title=f'Análisis de Rendimiento - {symbol}',
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        height=700
    )
    
    fig.update_xaxes(gridcolor=colors['grid'])
    fig.update_yaxes(gridcolor=colors['grid'])
    
    return html.Div([
        dcc.Graph(figure=fig),
        # Agregar métricas de rendimiento aquí
    ])