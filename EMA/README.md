# 📈 Sistema de Señales Forex con MT5

Un sistema automatizado de generación de señales de trading para el mercado Forex que utiliza MetaTrader 5 como fuente de datos y análisis técnico para identificar oportunidades de trading. se puede refinar mucho más, puedes porbar configurar los parametros a tu gusto y realizar el BackTesting antes de considerar emplearl en un escenario real.

## 🚀 Características

- **Análisis técnico automatizado** con indicadores RSI, Z-Score y ATR
- **Gestión de riesgo integrada** con cálculo automático de tamaño de posición
- **Múltiples pares de divisas** soportados (EUR/USD, GBP/USD, etc.)
- **Notificaciones por email** de señales generadas
- **Filtros de calidad** para evitar señales en condiciones de mercado desfavorables
- **Interfaz de línea de comandos** fácil de usar
- **Configuración flexible** mediante archivo JSON
- **Logging completo** para seguimiento y debugging

## 📋 Requisitos

### Software
- Python 3.7 o superior
- MetaTrader 5 instalado y configurado
- Cuenta de trading activa en un broker que soporte MT5

### Librerías de Python
```
MetaTrader5>=5.0.44
pandas>=1.3.0
numpy>=1.21.0
```

## 🛠️ Instalación

1. **Clona el repositorio:**
```bash
git clone https://github.com/tu-usuario/forex-signals.git
cd forex-signals
```

2. **Instala las dependencias:**
```bash
pip install -r requirements.txt
```

3. **Configura MetaTrader 5:**
   - Asegúrate de que MT5 esté instalado y funcionando
   - Habilita el trading algorítmico en MT5
   - Verifica que tengas una cuenta demo o real activa

4. **Ejecuta el programa por primera vez:**
```bash
python forex_signals.py
```
   - Se creará automáticamente un archivo `config.json` con la configuración por defecto

## ⚙️ Configuración

El sistema utiliza un archivo `config.json` para todas las configuraciones. Aquí están las secciones principales:

### Pares de Divisas
```json
"pairs": {
    "EURUSD": {"enabled": true, "min_spread": 3.0},
    "GBPUSD": {"enabled": true, "min_spread": 5.0},
    "AUDUSD": {"enabled": true, "min_spread": 4.0}
}
```

### Estrategia de Trading
```json
"strategy": {
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "zscore_window": 20,
    "zscore_threshold": 2.0,
    "atr_period": 14,
    "atr_multiplier": 1.5,
    "min_confidence": 0.6,
    "trend_filter": true,
    "trend_period": 200
}
```

### Gestión de Riesgo
```json
"risk": {
    "max_risk_per_trade": 0.02,  // 2% por operación
    "min_risk_reward": 1.5,      // Mínimo R/R de 1:1.5
    "max_spread_pips": 3.0,      // Spread máximo en pips
    "account_balance": 10000.0,   // Balance de cuenta
    "account_currency": "USD"     // Moneda de la cuenta
}
```

### Notificaciones por Email
```json
"notifications": {
    "email_enabled": true,
    "email_smtp": "smtp.gmail.com",
    "email_port": 587,
    "email_from": "tu_email@gmail.com",
    "email_password": "tu_app_password",
    "email_to": "destino@gmail.com"
}
```

## 🎯 Uso

### Modo Interactivo
Ejecuta el programa y selecciona una opción:

```bash
python forex_signals.py
```

**Opciones disponibles:**
1. **Escaneo único** - Analiza el mercado una vez y muestra señales
2. **Monitoreo continuo** - Ejecuta análisis cada X minutos según configuración
3. **Configurar parámetros** - Guía para editar config.json
4. **Calculadora de posición** - Herramienta para calcular tamaño de lote

### Ejemplo de Salida de Señal
```
=======================================
   NUEVA SEÑAL DE TRADING
=======================================
Par: EURUSD
Dirección: BUY
Entrada: 1.08450
Stop Loss: 1.08200
Take Profit: 1.08800
R/R: 1.40
Confianza: 75%

Tamaño de Posición:
  • Lotes: 0.80
  • Unidades: 80,000
  • Riesgo: $200.00 (2.0%)
  • Valor/pip: $8.00
  • Pips en riesgo: 25.0

Razón: Z-Score: -2.15, RSI: 28.3
Hora: 14:35:20
=======================================
```

## 📊 Estrategia de Trading

### Lógica de Señales
El sistema combina dos indicadores principales:

1. **Z-Score (Mean Reversion):**
   - Identifica cuando el precio se aleja significativamente de su media
   - Threshold: ±2.0 desviaciones estándar

2. **RSI (Relative Strength Index):**
   - Confirma condiciones de sobrecompra/sobreventa
   - Niveles: 30 (sobreventa) y 70 (sobrecompra)

### Condiciones de Entrada
- **Señal de COMPRA:** Z-Score < -2.0 Y RSI < 30
- **Señal de VENTA:** Z-Score > +2.0 Y RSI > 70

### Gestión de Riesgo
- **Stop Loss:** Basado en ATR (Average True Range)
- **Take Profit:** Retorno a la media móvil
- **Tamaño de posición:** Calculado automáticamente según riesgo por operación

## 📁 Estructura de Archivos

```
forex-signals/
│
├── forex_signals.py      # Programa principal
├── config.json          # Configuración (se crea automáticamente)
├── forex_signals.log    # Archivo de log
├── requirements.txt     # Dependencias de Python
└── README.md           # Este archivo
```

## 🔧 Personalización

### Modificar Indicadores
Edita la función `calculate_indicators()` para agregar nuevos indicadores:

```python
def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    # Tus indicadores personalizados aquí
    df['MACD'] = tu_calculo_macd(df)
    df['Bollinger'] = tu_calculo_bollinger(df)
    return df
```

### Cambiar Lógica de Señales
Modifica la función `check_trading_conditions()` para implementar tu estrategia:

```python
# Ejemplo: Señal basada en cruce de medias móviles
if df['SMA_Fast'].iloc[-1] > df['SMA_Slow'].iloc[-1]:
    # Lógica de señal de compra
    signal = TradingSignal(...)
```

## ⚠️ Consideraciones Importantes

### Riesgos
- **Este sistema es solo para fines educativos y de análisis**
- **Nunca operes con dinero que no puedas permitirte perder**
- **Siempre verifica las señales manualmente antes de ejecutar operaciones**
- **Los resultados pasados no garantizan resultados futuros**

### Limitaciones
- Depende de la conexión estable con MT5
- Los spreads pueden afectar la rentabilidad de las señales
- No incluye análisis fundamental ni eventos de noticias
- Requiere supervisión humana para mejores resultados

## 🐛 Solución de Problemas

### Error de conexión con MT5
```bash
Error inicializando MT5: (10004, 'No connection with the trade server')
```
**Solución:** Verifica que MT5 esté abierto y conectado a tu broker.

### Error de datos insuficientes
```bash
No se pudieron obtener datos para EURUSD
```
**Solución:** Asegúrate de que el símbolo esté disponible en tu broker y que tengas datos históricos.

### Problemas con emails
```bash
Error enviando email: (535, 'Authentication failed')
```
**Solución:** 
- Usa una "App Password" para Gmail en lugar de tu contraseña normal
- Habilita "Acceso de aplicaciones menos seguras" si es necesario

## 📈 Mejoras Futuras

- [ ] Interfaz web con dashboard en tiempo real
- [ ] Backtesting automatizado con datos históricos
- [ ] Integración con más brokers y plataformas
- [ ] Machine Learning para mejorar la precisión de señales
- [ ] API REST para integración con otras aplicaciones
- [ ] Notificaciones por Telegram y Discord
- [ ] Análisis de sentiment del mercado

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 📞 Contacto

**Autor:** Tu Nombre  
**Email:** tu.email@ejemplo.com  
**LinkedIn:** [tu-perfil](https://linkedin.com/in/tu-perfil)  
**GitHub:** [tu-usuario](https://github.com/tu-usuario)

## ⭐ Agradecimientos

- [MetaQuotes](https://www.mql5.com/) por la API de MetaTrader 5
- La comunidad de traders algorítmicos por compartir conocimiento
- Contribuidores y testers del proyecto

---

**⚡ Disclaimer:** Este software es solo para fines educativos. El trading de divisas conlleva riesgos significativos. Los desarrolladores no se hacen responsables por pérdidas financieras derivadas del uso de este sistema.
