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
                "EURUSD": {"enabled": True, "min_spread": 3.0},  # Aumentado para horario de cierre
                "GBPUSD": {"enabled": True, "min_spread": 5.0},  # Aumentado para horario de cierre
                "AUDUSD": {"enabled": True, "min_spread": 4.0},  # Aumentado para horario de cierre
                "NZDUSD": {"enabled": True, "min_spread": 5.0},  # Aumentado para horario de cierre
                "USDCHF": {"enabled": True, "min_spread": 4.0},  # Aumentado para horario de cierre
                "USDCAD": {"enabled": True, "min_spread": 5.0},  # Aumentado para horario de cierre
                "USDJPY": {"enabled": True, "min_spread": 3.0}   # Aumentado para horario de cierre
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
                "max_risk_per_trade": 0.02,
                "min_risk_reward": 1.5,
                "max_spread_pips": 3.0
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
    
    """########################"""    
    """seleccionar temporalidad"""
    """########################"""
    def get_market_data(self, symbol: str, timeframe=mt5.TIMEFRAME_H1, count: int = 500) -> Optional[pd.DataFrame]:
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
        
        # MT5 retorna: time, open, high, low, close, tick_volume, spread, real_volume
        # Mapear columnas correctamente según el número que tengamos
        if len(df.columns) == 7:
            # Formato estándar de MT5: open, high, low, close, tick_volume, spread, real_volume
            df.columns = ['Open', 'High', 'Low', 'Close', 'Tick_Volume', 'Spread', 'Real_Volume']
        elif len(df.columns) == 6:
            # Sin spread o sin real_volume
            df.columns = ['Open', 'High', 'Low', 'Close', 'Tick_Volume', 'Spread']
        elif len(df.columns) == 5:
            # Solo OHLCV básico
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        else:
            logging.warning(f"Número inesperado de columnas para {symbol}: {len(df.columns)}")
            # Asegurar que al menos tengamos las columnas básicas
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
        
        # Filtro de tendencia (opcional)
        if strategy_config['trend_filter']:
            df['SMA_Trend'] = df['Close'].rolling(strategy_config['trend_period']).mean()
            df['Trend'] = np.where(df['Close'] > df['SMA_Trend'], 1, -1)
        
        return df
    
    def is_market_open(self) -> bool:
        """Verificar si el mercado Forex está abierto"""
        now = datetime.now()
        
        # Forex está cerrado los fines de semana (sábado y domingo)
        if now.weekday() >= 5:  # 5=sábado, 6=domingo
            return False
            
        # Forex cierra el viernes a las 22:00 GMT y abre el domingo a las 22:00 GMT
        # Ajustar según tu zona horaria local
        
        return True  # Simplificado - puedes agregar lógica más compleja aquí
    
    def check_trading_conditions(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Verificar condiciones de trading y generar señal"""
        if len(df) < 50:  # Datos insuficientes
            return None
        
        current = df.iloc[-1]
        strategy = self.config['strategy']
        risk = self.config['risk']
        
        # Verificar si el mercado está abierto
        if not self.is_market_open():
            logging.info("Mercado cerrado - spreads pueden estar elevados")
            return None
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return None
        
        # Calcular spread en pips de manera más robusta
        point = symbol_info.point
        digits = symbol_info.digits
        
        # Determinar el multiplicador de pips basado en los dígitos
        if 'JPY' in symbol:
            # Para pares JPY (ej: USDJPY) - pip es 0.01
            pip_multiplier = 100 if digits == 3 else 1
        else:
            # Para pares major (ej: EURUSD) - pip es 0.0001  
            pip_multiplier = 10000 if digits == 5 else 1
        
        spread_pips = symbol_info.spread / pip_multiplier
        max_spread = self.config['pairs'].get(symbol, {}).get('min_spread', risk['max_spread_pips'])
        
        if spread_pips > max_spread:
            logging.info(f"{symbol}: Spread muy alto ({spread_pips:.1f} pips)")
            return None
        
        # Verificar que tenemos todos los indicadores necesarios
        required_indicators = ['ZScore', 'RSI', 'ATR', 'SMA']
        for indicator in required_indicators:
            if indicator not in df.columns or pd.isna(current[indicator]):
                logging.warning(f"{symbol}: Falta indicador {indicator} o es NaN")
                return None
        
        # Condiciones de entrada
        signal = None
        confidence = 0.0
        reason = ""
        
        # Señal de COMPRA (Long)
        if (current['ZScore'] < -strategy['zscore_threshold'] and 
            current['RSI'] < strategy['rsi_oversold']):
            
            confidence = min(0.9, abs(current['ZScore']) / 3.0 + (80 - current['RSI']) / 100)
            
            # Verificar filtro de tendencia
            if strategy['trend_filter'] and 'Trend' in df.columns and current['Trend'] < 0:
                confidence *= 0.7  # Reducir confianza si va contra tendencia
            
            # Calcular precios
            entry_price = symbol_info.ask
            atr = current['ATR']
            stop_loss = entry_price - (atr * strategy['atr_multiplier'])
            take_profit = current['SMA']  # Retorno a la media
            
            # Verificar risk/reward
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
        
        # Señal de VENTA (Short)
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
        
        # Verificar confianza mínima
        if signal and signal.confidence >= strategy['min_confidence']:
            return signal
        
        return None
    
    def is_duplicate_signal(self, signal: TradingSignal) -> bool:
        """Verificar si la señal es duplicada (evitar spam)"""
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
            
            body = f"""
            NUEVA SEÑAL DE TRADING
            
            Par: {signal.pair}
            Dirección: {signal.direction}
            Precio de Entrada: {signal.entry_price:.5f}
            Stop Loss: {signal.stop_loss:.5f}
            Take Profit: {signal.take_profit:.5f}
            Risk/Reward: {signal.risk_reward:.2f}
            Confianza: {signal.confidence:.0%}
            
            Razón: {signal.reason}
            Hora: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            Esta es solo una señal. Analiza el mercado antes de operar.
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
        return f"""
=======================================
   NUEVA SEÑAL DE TRADING
=======================================
Par: {signal.pair}
Dirección: {signal.direction}
Entrada: {signal.entry_price:.5f}
Stop Loss: {signal.stop_loss:.5f}
Take Profit: {signal.take_profit:.5f}
R/R: {signal.risk_reward:.2f}
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
        
        signals = self.scan_markets()
        
        if signals:
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
                time.sleep(60)  # Esperar 1 minuto antes de reintentar

def main():
    """Función principal"""
    signal_generator = ForexSignalGenerator()
    
    # Opciones de ejecución
    print("Sistema de Señales Forex")
    print("1. Escaneo único")
    print("2. Monitoreo continuo")
    print("3. Configurar parámetros")
    
    choice = input("\nSelecciona una opción (1-3): ").strip()
    
    if choice == '1':
        signal_generator.run_once()
    elif choice == '2':
        signal_generator.run_continuously()
    elif choice == '3':
        print("\nEdita el archivo 'config.json' para cambiar parámetros")
        print("Reinicia el programa después de los cambios")
    else:
        print("Opción no válida")

if __name__ == "__main__":
    main()