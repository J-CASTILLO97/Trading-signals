import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging
import sys

# Configurar codificación UTF-8 para Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Configurar logging sin emojis para evitar errores de codificación
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forex_signals.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

@dataclass
class TradingSignal:
    """Clase para representar una señal de trading"""
    pair: str
    direction: str  # 'BUY' o 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    confidence: float
    timestamp: datetime
    reason: str
    position_size: float = 0.0  # Tamaño de posición en lotes
    risk_amount: float = 0.0   # Monto en riesgo en dinero

class ForexSignalGenerator:
    def __init__(self, config_file: str = 'config.json'):
        """Inicializar el generador de señales"""
        self.config = self.load_config(config_file)
        self.mt5_connected = False
        self.last_signals = {}  # Para evitar señales duplicadas
        
    def load_config(self, config_file: str) -> dict:
        """Cargar configuración desde archivo JSON"""
        default_config = {
            "pairs": {
                "EURUSD": {"enabled": True, "min_spread": 3.0},
                "GBPUSD": {"enabled": True, "min_spread": 5.0},
                "AUDUSD": {"enabled": True, "min_spread": 4.0},
                "NZDUSD": {"enabled": True, "min_spread": 5.0},
                "USDCHF": {"enabled": True, "min_spread": 4.0},
                "USDCAD": {"enabled": True, "min_spread": 5.0},
                "USDJPY": {"enabled": True, "min_spread": 3.0},
                "XAUUSD": {"enabled": True, "min_spread": 5.0}
            },
            "strategy": {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "zscore_window": 20,
                "zscore_threshold": 2.0,
                "atr_period": 14,
                "atr_multiplier": 1.5,
                "min_confidence": 0.6,
                "trend_filter": True,
                "trend_period": 200
            },
            "risk": {
                "max_risk_per_trade": 0.02,        # 2% del balance por operación
                "min_risk_reward": 1.5,
                "max_spread_pips": 3.0,
                "use_equity_for_risk": True,       # Usar equity en lugar de balance
                "max_position_size": 10.0,         # Máximo de lotes por operación
                "min_account_balance": 1000        # Balance mínimo para operar
            },
            "notifications": {
                "email_enabled": True,
                "email_smtp": "smtp.gmail.com",
                "email_port": 587,
                "email_from": "tu_email@gmail.com",
                "email_password": "tu_app_password",
                "email_to": "tu_email@gmail.com",
                "telegram_enabled": False,
                "telegram_token": "",
                "telegram_chat_id": ""
            },
            "schedule": {
                "check_interval_minutes": 30,
                "trading_hours": {
                    "start": "08:00",
                    "end": "17:00"
                },
                "avoid_news_times": True
            }
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
        else:
            # Crear archivo de configuración por defecto
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            logging.info(f"Archivo de configuración creado: {config_file}")
            return default_config
    
    def connect_mt5(self) -> bool:
        """Conectar a MetaTrader 5"""
        if not mt5.initialize():
            logging.error(f"Error inicializando MT5: {mt5.last_error()}")
            return False
        
        # Verificar conexión
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("No se pudo obtener información de la cuenta")
            return False
        
        logging.info(f"Conectado a MT5 - Cuenta: {account_info.login}, Broker: {account_info.company}")
        self.mt5_connected = True
        return True
    
    """======================================="""
    """   Seleccionar TimeFrame de análisis   """
    """======================================="""

    def get_market_data(self, symbol: str, timeframe=mt5.TIMEFRAME_M5, count: int = 500) -> Optional[pd.DataFrame]:
        """Obtener datos de mercado desde MT5"""
        if not self.mt5_connected:
            if not self.connect_mt5():
                return None
        
        # Obtener datos históricos
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            logging.error(f"No se pudieron obtener datos para {symbol}: {mt5.last_error()}")
            return None
        
        # Convertir a DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Mapear columnas correctamente
        if len(df.columns) == 7:
            df.columns = ['Open', 'High', 'Low', 'Close', 'Tick_Volume', 'Spread', 'Real_Volume']
        elif len(df.columns) == 6:
            df.columns = ['Open', 'High', 'Low', 'Close', 'Tick_Volume', 'Spread']
        elif len(df.columns) == 5:
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        else:
            logging.warning(f"Número inesperado de columnas para {symbol}: {len(df.columns)}")
            base_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            extra_cols = [f'Extra_{i}' for i in range(len(df.columns) - 5)]
            df.columns = base_cols + extra_cols
        
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular indicadores técnicos"""
        strategy_config = self.config['strategy']
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=strategy_config['rsi_period']).mean()
        avg_loss = loss.rolling(window=strategy_config['rsi_period']).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Z-Score (Mean Reversion)
        window = strategy_config['zscore_window']
        df['SMA'] = df['Close'].rolling(window).mean()
        df['STD'] = df['Close'].rolling(window).std()
        df['ZScore'] = (df['Close'] - df['SMA']) / (df['STD'] + 1e-10)
        
        # ATR para stop loss
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(strategy_config['atr_period']).mean()
        
        # Filtro de tendencia
        if strategy_config['trend_filter']:
            df['SMA_Trend'] = df['Close'].rolling(strategy_config['trend_period']).mean()
            df['Trend'] = np.where(df['Close'] > df['SMA_Trend'], 1, -1)
        
        return df
    
    def calculate_position_size(self, signal: TradingSignal) -> TradingSignal:
        """Calcular el tamaño de posición basado en el riesgo configurado"""
        try:
            # Obtener información de la cuenta
            account_info = mt5.account_info()
            if account_info is None:
                logging.error("No se pudo obtener información de la cuenta")
                return signal
            
            # Verificar balance mínimo
            balance = account_info.balance
            equity = account_info.equity
            
            if balance < self.config['risk']['min_account_balance']:
                logging.warning(f"Balance insuficiente: ${balance:.2f}")
                return signal
            
            # Usar equity o balance según configuración
            account_value = equity if self.config['risk']['use_equity_for_risk'] else balance
            
            # Obtener información del símbolo
            symbol_info = mt5.symbol_info(signal.pair)
            if symbol_info is None:
                logging.error(f"No se pudo obtener información del símbolo {signal.pair}")
                return signal
            
            # Calcular el riesgo en dinero
            risk_percentage = self.config['risk']['max_risk_per_trade']
            risk_amount = account_value * risk_percentage
            
            # Calcular la distancia del stop loss
            if signal.direction == 'BUY':
                risk_distance = signal.entry_price - signal.stop_loss
            else:  # SELL
                risk_distance = signal.stop_loss - signal.entry_price
            
            # Determinar el valor del pip por lote
            contract_size = symbol_info.trade_contract_size
            
            if 'JPY' in signal.pair:
                # Para pares JPY (pip = 0.01)
                pip_size = 0.01
                if signal.pair.startswith('USD'):
                    # USDJPY: pip value = (pip_size / price) * contract_size
                    pip_value_per_lot = (pip_size / signal.entry_price) * contract_size
                else:
                    # EURJPY, GBPJPY: necesitaríamos conversión, usamos aproximación
                    pip_value_per_lot = pip_size * contract_size / signal.entry_price
            else:
                # Para pares major (pip = 0.0001)
                pip_size = 0.0001
                if signal.pair.endswith('USD'):
                    # Pares XXX/USD: pip value = pip_size * contract_size
                    pip_value_per_lot = pip_size * contract_size
                else:
                    # USD/XXX: pip value = (pip_size / price) * contract_size
                    pip_value_per_lot = (pip_size / signal.entry_price) * contract_size
            
            # Calcular pips en riesgo
            pips_at_risk = risk_distance / pip_size
            
            # Calcular tamaño de posición
            if pips_at_risk > 0 and pip_value_per_lot > 0:
                # Fórmula: Tamaño = Riesgo en dinero / (Pips en riesgo * Valor del pip por lote)
                position_size = risk_amount / (pips_at_risk * pip_value_per_lot)
                
                # Ajustar a los límites del broker
                min_lot = symbol_info.volume_min
                max_lot = min(symbol_info.volume_max, self.config['risk']['max_position_size'])
                lot_step = symbol_info.volume_step
                
                # Redondear al step permitido
                position_size = round(position_size / lot_step) * lot_step
                
                # Aplicar límites
                position_size = max(min_lot, min(position_size, max_lot))
                
                # Actualizar la señal
                signal.position_size = position_size
                signal.risk_amount = risk_amount
                
                logging.info(f"Tamaño calculado para {signal.pair}: {position_size} lotes "
                            f"(Riesgo: ${risk_amount:.2f}, {pips_at_risk:.1f} pips)")
            else:
                logging.error(f"Error en cálculo: pips_at_risk={pips_at_risk:.1f}, "
                             f"pip_value={pip_value_per_lot:.2f}")
        
        except Exception as e:
            logging.error(f"Error calculando tamaño de posición: {e}")
        
        return signal
    
    def is_market_open(self) -> bool:
        """Verificar si el mercado Forex está abierto"""
        now = datetime.now()
        
        # Forex está cerrado los fines de semana
        if now.weekday() >= 5:  # 5=sábado, 6=domingo
            return False
            
        return True
    
    def check_trading_conditions(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Verificar condiciones de trading y generar señal"""
        if len(df) < 50:
            return None
        
        current = df.iloc[-1]
        strategy = self.config['strategy']
        risk = self.config['risk']
        
        # Verificar mercado abierto
        if not self.is_market_open():
            return None
            
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return None
        
        # Verificar spread
        point = symbol_info.point
        digits = symbol_info.digits
        
        if 'JPY' in symbol:
            pip_multiplier = 100 if digits == 3 else 1
        else:
            pip_multiplier = 10000 if digits == 5 else 1
        
        spread_pips = symbol_info.spread / pip_multiplier
        max_spread = self.config['pairs'].get(symbol, {}).get('min_spread', risk['max_spread_pips'])
        
        if spread_pips > max_spread:
            logging.info(f"{symbol}: Spread muy alto ({spread_pips:.1f} pips)")
            return None
        
        # Verificar indicadores
        required_indicators = ['ZScore', 'RSI', 'ATR', 'SMA']
        for indicator in required_indicators:
            if indicator not in df.columns or pd.isna(current[indicator]):
                logging.warning(f"{symbol}: Falta indicador {indicator}")
                return None
        
        signal = None
        
        # Señal de COMPRA
        if (current['ZScore'] < -strategy['zscore_threshold'] and 
            current['RSI'] < strategy['rsi_oversold']):
            
            confidence = min(0.9, abs(current['ZScore']) / 3.0 + (80 - current['RSI']) / 100)
            
            if strategy['trend_filter'] and 'Trend' in df.columns and current['Trend'] < 0:
                confidence *= 0.7
            
            entry_price = symbol_info.ask
            atr = current['ATR']
            stop_loss = entry_price - (atr * strategy['atr_multiplier'])
            take_profit = current['SMA']
            
            risk_distance = entry_price - stop_loss
            reward_distance = take_profit - entry_price
            risk_reward = reward_distance / risk_distance if risk_distance > 0 else 0
            
            if risk_reward >= risk['min_risk_reward']:
                reason = f"Z-Score: {current['ZScore']:.2f}, RSI: {current['RSI']:.1f}"
                
                signal = TradingSignal(
                    pair=symbol,
                    direction='BUY',
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward=risk_reward,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    reason=reason
                )
        
        # Señal de VENTA
        elif (current['ZScore'] > strategy['zscore_threshold'] and 
              current['RSI'] > strategy['rsi_overbought']):
            
            confidence = min(0.9, abs(current['ZScore']) / 3.0 + (current['RSI'] - 20) / 100)
            
            if strategy['trend_filter'] and 'Trend' in df.columns and current['Trend'] > 0:
                confidence *= 0.7
            
            entry_price = symbol_info.bid
            atr = current['ATR']
            stop_loss = entry_price + (atr * strategy['atr_multiplier'])
            take_profit = current['SMA']
            
            risk_distance = stop_loss - entry_price
            reward_distance = entry_price - take_profit
            risk_reward = reward_distance / risk_distance if risk_distance > 0 else 0
            
            if risk_reward >= risk['min_risk_reward']:
                reason = f"Z-Score: {current['ZScore']:.2f}, RSI: {current['RSI']:.1f}"
                
                signal = TradingSignal(
                    pair=symbol,
                    direction='SELL',
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward=risk_reward,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    reason=reason
                )
        
        # Calcular tamaño de posición si hay señal
        if signal and signal.confidence >= strategy['min_confidence']:
            signal = self.calculate_position_size(signal)
            return signal
        
        return None
    
    def is_duplicate_signal(self, signal: TradingSignal) -> bool:
        """Verificar si la señal es duplicada"""
        key = f"{signal.pair}_{signal.direction}"
        
        if key in self.last_signals:
            last_time = self.last_signals[key]
            if (datetime.now() - last_time).seconds < 3600:  # 1 hora
                return True
        
        self.last_signals[key] = datetime.now()
        return False
    
    def send_email_notification(self, signal: TradingSignal):
        """Enviar notificación por email"""
        if not self.config['notifications']['email_enabled']:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['notifications']['email_from']
            msg['To'] = self.config['notifications']['email_to']
            msg['Subject'] = f"SEÑAL FOREX: {signal.direction} {signal.pair}"
            
            # Obtener información de la cuenta
            account_info = mt5.account_info()
            balance = account_info.balance if account_info else 0
            equity = account_info.equity if account_info else 0
            
            # Calcular potencial ganancia/pérdida
            if signal.direction == 'BUY':
                potential_profit = (signal.take_profit - signal.entry_price) * signal.position_size * 100000
                potential_loss = (signal.entry_price - signal.stop_loss) * signal.position_size * 100000
            else:
                potential_profit = (signal.entry_price - signal.take_profit) * signal.position_size * 100000
                potential_loss = (signal.stop_loss - signal.entry_price) * signal.position_size * 100000
            
            # Ajustar para pares JPY
            if 'JPY' in signal.pair:
                potential_profit /= 100
                potential_loss /= 100
            
            body = f"""
            🔔 NUEVA SEÑAL DE TRADING 🔔
            
            📊 INFORMACIÓN DEL PAR:
            Par: {signal.pair}
            Dirección: {signal.direction} {'📈' if signal.direction == 'BUY' else '📉'}
            Precio de Entrada: {signal.entry_price:.5f}
            Stop Loss: {signal.stop_loss:.5f}
            Take Profit: {signal.take_profit:.5f}
            
            💰 GESTIÓN DE RIESGO:
            Tamaño de Posición: {signal.position_size:.2f} lotes
            Riesgo Asumido: ${signal.risk_amount:.2f} ({self.config['risk']['max_risk_per_trade']:.1%} del capital)
            Potencial Ganancia: ${potential_profit:.2f}
            Potencial Pérdida: ${potential_loss:.2f}
            Risk/Reward Ratio: {signal.risk_reward:.2f}
            
            📈 INFORMACIÓN DE CUENTA:
            Balance: ${balance:.2f}
            Equity: ${equity:.2f}
            
            🎯 ANÁLISIS:
            Confianza: {signal.confidence:.0%}
            Razón: {signal.reason}
            Timestamp: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            ⚠️ DISCLAIMER:
            Esta es solo una señal automatizada basada en análisis técnico.
            Siempre verifica las condiciones del mercado antes de operar.
            El trading conlleva riesgos y puedes perder tu capital.
            
            📱 Para mejores resultados:
            1. Verifica las noticias económicas
            2. Confirma la dirección de la tendencia
            3. Considera las condiciones del mercado
            4. Nunca arriesgues más de lo que puedes permitirte perder
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(
                self.config['notifications']['email_smtp'],
                self.config['notifications']['email_port']
            )
            server.starttls()
            server.login(
                self.config['notifications']['email_from'],
                self.config['notifications']['email_password']
            )
            text = msg.as_string()
            server.sendmail(
                self.config['notifications']['email_from'],
                self.config['notifications']['email_to'],
                text
            )
            server.quit()
            
            logging.info(f"Email enviado para señal {signal.pair} {signal.direction}")
            
        except Exception as e:
            logging.error(f"Error enviando email: {e}")
    
    def format_signal_message(self, signal: TradingSignal) -> str:
        """Formatear mensaje de señal para consola"""
        # Obtener información de cuenta
        account_info = mt5.account_info()
        balance = account_info.balance if account_info else 0
        equity = account_info.equity if account_info else 0
        
        # Calcular potencial ganancia/pérdida
        if signal.direction == 'BUY':
            potential_profit = (signal.take_profit - signal.entry_price) * signal.position_size * 100000
            potential_loss = (signal.entry_price - signal.stop_loss) * signal.position_size * 100000
        else:
            potential_profit = (signal.entry_price - signal.take_profit) * signal.position_size * 100000
            potential_loss = (signal.stop_loss - signal.entry_price) * signal.position_size * 100000
        
        # Ajustar para pares JPY
        if 'JPY' in signal.pair:
            potential_profit /= 100
            potential_loss /= 100
        
        return f"""
=======================================
        NUEVA SEÑAL DE TRADING
=======================================
Par: {signal.pair}
Dirección: {signal.direction} {'↗️' if signal.direction == 'BUY' else '↘️'}
Entrada: {signal.entry_price:.5f}
Stop Loss: {signal.stop_loss:.5f}
Take Profit: {signal.take_profit:.5f}

💰 GESTIÓN DE RIESGO:
Tamaño: {signal.position_size:.2f} lotes
Riesgo: ${signal.risk_amount:.2f} ({self.config['risk']['max_risk_per_trade']:.1%})
Pot. Ganancia: ${potential_profit:.2f}
Pot. Pérdida: ${potential_loss:.2f}
R/R Ratio: {signal.risk_reward:.2f}

📊 CUENTA:
Balance: ${balance:.2f}
Equity: ${equity:.2f}

🎯 ANÁLISIS:
Confianza: {signal.confidence:.0%}
Razón: {signal.reason}
Hora: {signal.timestamp.strftime('%H:%M:%S')}
=======================================
        """
    
    def scan_markets(self) -> List[TradingSignal]:
        """Escanear todos los pares configurados"""
        signals = []
        
        for symbol, settings in self.config['pairs'].items():
            if not settings.get('enabled', True):
                continue
            
            try:
                # Obtener datos
                df = self.get_market_data(symbol)
                if df is None:
                    logging.warning(f"No se pudieron obtener datos para {symbol}")
                    continue
                
                # Calcular indicadores
                df = self.calculate_indicators(df)
                
                # Verificar señal
                signal = self.check_trading_conditions(symbol, df)
                
                if signal and not self.is_duplicate_signal(signal):
                    signals.append(signal)
                    logging.info(f"Señal encontrada: {signal.pair} {signal.direction}")
                    
            except Exception as e:
                logging.error(f"Error procesando {symbol}: {e}")
        
        return signals
    
    def run_once(self):
        """Ejecutar un escaneo completo"""
        logging.info("Iniciando escaneo de mercados...")
        
        if not self.mt5_connected:
            if not self.connect_mt5():
                logging.error("No se pudo conectar a MT5")
                return
        
        # Verificar balance mínimo
        account_info = mt5.account_info()
        if account_info and account_info.balance < self.config['risk']['min_account_balance']:
            logging.error(f"Balance insuficiente: ${account_info.balance:.2f}")
            return
        
        signals = self.scan_markets()
        
        if signals:
            logging.info(f"Se encontraron {len(signals)} señal(es)")
            for signal in signals:
                print(self.format_signal_message(signal))
                self.send_email_notification(signal)
        else:
            logging.info("No se encontraron señales en este momento")
    
    def run_continuously(self):
        """Ejecutar continuamente según configuración"""
        interval = self.config['schedule']['check_interval_minutes']
        
        logging.info(f"Iniciando monitoreo continuo (cada {interval} minutos)")
        
        while True:
            try:
                # Verificar horario de trading
                now = datetime.now().time()
                start_time = datetime.strptime(
                    self.config['schedule']['trading_hours']['start'], '%H:%M'
                ).time()
                end_time = datetime.strptime(
                    self.config['schedule']['trading_hours']['end'], '%H:%M'
                ).time()
                
                if start_time <= now <= end_time:
                    self.run_once()
                else:
                    logging.info(f"Fuera del horario de trading ({start_time} - {end_time})")
                
                # Esperar siguiente intervalo
                time.sleep(interval * 60)
                
            except KeyboardInterrupt:
                logging.info("Deteniendo monitoreo...")
                break
            except Exception as e:
                logging.error(f"Error en el bucle principal: {e}")
                time.sleep(60)
    
    def get_account_summary(self):
        """Mostrar resumen de la cuenta"""
        if not self.mt5_connected:
            if not self.connect_mt5():
                return
        
        account_info = mt5.account_info()
        if account_info is None:
            print("No se pudo obtener información de la cuenta")
            return
        
        print(f"""
=======================================
        RESUMEN DE CUENTA
=======================================
Cuenta: {account_info.login}
Broker: {account_info.company}
Balance: ${account_info.balance:.2f}
Equity: ${account_info.equity:.2f}
Margen Libre: ${account_info.margin_free:.2f}
Margen Usado: ${account_info.margin:.2f}
Nivel de Margen: {account_info.margin_level:.2f}%

Configuración de Riesgo:
- Riesgo por operación: {self.config['risk']['max_risk_per_trade']:.1%}
- Monto en riesgo: ${account_info.balance * self.config['risk']['max_risk_per_trade']:.2f}
- R/R mínimo: {self.config['risk']['min_risk_reward']:.1f}
=======================================
        """)

def main():
    """Función principal"""
    signal_generator = ForexSignalGenerator()
    
    # Opciones de ejecución
    print("🚀 Sistema de Señales Forex con Gestión de Riesgo")
    print("=" * 50)
    print("1. Escaneo único")
    print("2. Monitoreo continuo")
    print("3. Ver resumen de cuenta")
    print("4. Configurar parámetros")
    print("5. Salir")
    
    while True:
        choice = input("\n📝 Selecciona una opción (1-5): ").strip()
        
        if choice == '1':
            signal_generator.run_once()
        elif choice == '2':
            print("🔄 Iniciando monitoreo continuo...")
            print("💡 Presiona Ctrl+C para detener")
            signal_generator.run_continuously()
        elif choice == '3':
            signal_generator.get_account_summary()
        elif choice == '4':
            print("\n⚙️ Para configurar parámetros:")
            print("1. Edita el archivo 'config.json'")
            print("2. Ajusta los valores según tus preferencias")
            print("3. Reinicia el programa")
            print("\n📋 Parámetros principales:")
            print("- max_risk_per_trade: Porcentaje de riesgo por operación")
            print("- min_risk_reward: Relación mínima riesgo/beneficio")
            print("- max_position_size: Tamaño máximo de posición en lotes")
            print("- pairs: Pares a monitorear y sus configuraciones")
        elif choice == '5':
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción no válida. Por favor selecciona 1-5.")

if __name__ == "__main__":
    main()