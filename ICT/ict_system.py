import pandas as pd
import numpy as np
from datetime import datetime, time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os
import schedule
import time as time_module
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ict_trading.log'),
        logging.StreamHandler()
    ]
)

# Intentar importar MetaTrader5, si no está disponible usar modo simulado
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    print("✅ MetaTrader5 disponible")
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 no disponible - usando modo simulado")

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class MarketStructure(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"

@dataclass
class TradingSignal:
    timestamp: datetime
    pair: str
    signal: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    confidence: float
    strategy: str
    spread_adjusted: bool
    position_size: float = 0.0
    risk_amount: float = 0.0

class TechnicalIndicators:
    """Implementación manual de indicadores técnicos"""
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

class ConfigManager:
    """Gestor de configuración para archivos JSON"""
    
    @staticmethod
    def create_default_config():
        """Crear archivo de configuración por defecto"""
        default_config = {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "port": 587,
                "email": "",
                "password": "",
                "recipient": ""
            },
            "mt5": {
                "auto_connect": True,
                "login": None,
                "password": "",
                "server": ""
            },
            "trading": {
                "default_balance": 10000,
                "risk_percentage": 1.0,
                "default_symbols": ["EURUSD", "XAUUSD", "GBPUSD"],
                "default_interval": 5
            }
        }
        
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4, ensure_ascii=False)
        
        print("📁 Archivo config.json creado con configuración por defecto")
        print("📝 Edite config.json con sus datos de email y MT5")
        return default_config
    
    @staticmethod
    def load_config():
        """Cargar configuración desde archivo JSON"""
        try:
            if not os.path.exists('config.json'):
                print("📁 No se encontró config.json, creando archivo por defecto...")
                return ConfigManager.create_default_config()
            
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print("✅ Configuración cargada desde config.json")
            return config
            
        except Exception as e:
            logging.error(f"Error cargando configuración: {e}")
            print("⚠️ Error en config.json, creando nueva configuración...")
            return ConfigManager.create_default_config()
    
    @staticmethod
    def save_config(config):
        """Guardar configuración en archivo JSON"""
        try:
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print("✅ Configuración guardada en config.json")
        except Exception as e:
            logging.error(f"Error guardando configuración: {e}")

class EmailConfig:
    def __init__(self, smtp_server: str, port: int, email: str, password: str):
        self.smtp_server = smtp_server
        self.port = port
        self.email = email
        self.password = password

class MetaTraderConnection:
    def __init__(self):
        self.connected = False
        self.account_info = None
        self.available = MT5_AVAILABLE
        self.terminal_info = None
    
    def connect(self, login: int = None, password: str = None, server: str = None, auto_detect: bool = True):
        """Conectar a MetaTrader 5 con detección automática mejorada"""
        if not self.available:
            logging.warning("MetaTrader5 no está disponible")
            return False
        
        try:
            print("🔄 Iniciando conexión a MetaTrader 5...")
            
            # Intentar inicializar MT5
            if not mt5.initialize():
                error = mt5.last_error()
                logging.error(f"Error al inicializar MT5: {error}")
                
                # Intentar con ruta específica si falla
                possible_paths = [
                    "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
                    "C:\\Program Files (x86)\\MetaTrader 5\\terminal64.exe",
                    "C:\\Users\\{}\\AppData\\Roaming\\MetaQuotes\\Terminal\\*.exe".format(os.getlogin())
                ]
                
                for path in possible_paths:
                    if "*" not in path and os.path.exists(path):
                        print(f"🔄 Intentando inicializar MT5 desde: {path}")
                        if mt5.initialize(path):
                            print("✅ MT5 inicializado desde ruta específica")
                            break
                else:
                    print("❌ No se pudo inicializar MT5. Asegúrese de que esté instalado y accesible.")
                    return False
            
            # Obtener información del terminal
            self.terminal_info = mt5.terminal_info()
            if self.terminal_info:
                print(f"📊 Terminal MT5: {self.terminal_info.name}")
                print(f"📊 Versión: {self.terminal_info.build}")
                print(f"📊 Path: {self.terminal_info.path}")
                print(f"📊 Estado: {'Conectado' if self.terminal_info.connected else 'Desconectado'}")
            
            # Intentar login si se proporcionan credenciales
            if login and password and server:
                print(f"🔐 Intentando login con cuenta {login} en servidor {server}...")
                if not mt5.login(login, password, server):
                    error = mt5.last_error()
                    logging.error(f"Error al hacer login: {error}")
                    print(f"❌ Error de login: {error}")
                    # Continuar sin login específico para usar cuenta por defecto
            
            # Obtener información de la cuenta
            self.account_info = mt5.account_info()
            if self.account_info is None:
                logging.error("Error al obtener información de cuenta")
                print("⚠️ No se pudo obtener información de la cuenta")
                print("💡 MT5 está conectado pero sin cuenta activa")
                print("💡 Abra MT5 manualmente y conéctese a su cuenta")
                return False
            
            self.connected = True
            print("✅ Conexión exitosa a MetaTrader 5")
            print(f"💰 Cuenta: {self.account_info.login}")
            print(f"💰 Servidor: {self.account_info.server}")
            print(f"💰 Balance: ${self.account_info.balance:,.2f}")
            print(f"💰 Equity: ${self.account_info.equity:,.2f}")
            print(f"💰 Moneda: {self.account_info.currency}")
            
            # Verificar símbolos disponibles
            symbols = mt5.symbols_get()
            if symbols:
                print(f"📊 Símbolos disponibles: {len(symbols)}")
                forex_symbols = [s.name for s in symbols if any(pair in s.name for pair in ['EUR', 'GBP', 'USD', 'JPY'])]
                print(f"📊 Símbolos Forex: {len(forex_symbols)}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error en conexión MT5: {e}")
            print(f"❌ Error en conexión: {e}")
            return False
    
    def get_symbol_data(self, symbol: str, timeframe: int, count: int = 1000) -> pd.DataFrame:
        """Obtener datos históricos de un símbolo"""
        if not self.available or not self.connected:
            return None
        
        try:
            print(f"📊 Obteniendo datos de {symbol} (últimas {count} velas)...")
            
            # Verificar que el símbolo esté disponible
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"⚠️ Símbolo {symbol} no encontrado. Intentando variaciones...")
                
                # Intentar variaciones del símbolo
                variations = [
                    symbol,
                    symbol + "m",  # Micro lotes
                    symbol + ".m", # Algunas variaciones
                    symbol[:-3] + "/" + symbol[-3:],  # Con slash
                ]
                
                for var in variations:
                    symbol_info = mt5.symbol_info(var)
                    if symbol_info:
                        symbol = var
                        print(f"✅ Usando símbolo: {symbol}")
                        break
                else:
                    logging.error(f"Símbolo {symbol} no disponible en este broker")
                    return None
            
            # Asegurar que el símbolo esté visible en Market Watch
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    logging.warning(f"No se pudo activar símbolo {symbol}")
            
            # Obtener datos históricos
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                error = mt5.last_error()
                logging.error(f"Error al obtener datos de {symbol}: {error}")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.rename(columns={
                'time': 'timestamp',
                'tick_volume': 'volume'
            }, inplace=True)
            
            print(f"✅ Obtenidos {len(df)} registros de {symbol}")
            print(f"📊 Rango: {df['timestamp'].min()} a {df['timestamp'].max()}")
            print(f"📊 Último precio: {df['close'].iloc[-1]:.5f}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error obteniendo datos de {symbol}: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Obtener información del símbolo (spread, point, etc.)"""
        if not self.available or not self.connected:
            return None
        
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None
            
            # Obtener tick actual para spread en tiempo real
            tick = mt5.symbol_info_tick(symbol)
            current_spread = 0
            if tick:
                current_spread = tick.ask - tick.bid
            
            symbol_data = {
                'spread': current_spread,
                'point': info.point,
                'digits': info.digits,
                'trade_contract_size': info.trade_contract_size,
                'volume_min': info.volume_min,
                'volume_step': info.volume_step,
                'bid': info.bid,
                'ask': info.ask,
                'currency_base': info.currency_base,
                'currency_profit': info.currency_profit,
                'margin_initial': info.margin_initial
            }
            
            print(f"📊 {symbol} - Spread actual: {current_spread:.5f} ({current_spread/info.point:.1f} pips)")
            
            return symbol_data
            
        except Exception as e:
            logging.error(f"Error obteniendo info de {symbol}: {e}")
            return None
    
    def get_balance(self) -> float:
        """Obtener balance actual de la cuenta"""
        if not self.available or not self.connected or not self.account_info:
            return 0.0
        
        try:
            # Actualizar información de cuenta
            self.account_info = mt5.account_info()
            return self.account_info.balance if self.account_info else 0.0
        except:
            return 0.0
    
    def disconnect(self):
        """Desconectar de MetaTrader 5"""
        if self.available:
            mt5.shutdown()
        self.connected = False
        logging.info("Desconectado de MetaTrader 5")

class PositionSizeCalculator:
    def __init__(self, risk_percentage: float = 1.0):
        self.risk_percentage = risk_percentage / 100.0  # Convertir a decimal
    
    def calculate_position_size(self, balance: float, entry_price: float, 
                              stop_loss: float, symbol_info: Dict) -> Tuple[float, float]:
        """
        Calcula el tamaño de posición basado en riesgo del 1%
        Retorna: (position_size, risk_amount)
        """
        try:
            # Cantidad de riesgo en dinero
            risk_amount = balance * self.risk_percentage
            
            # Diferencia en pips/puntos entre entrada y stop loss
            risk_in_points = abs(entry_price - stop_loss)
            
            # Para Forex
            if symbol_info.get('point', 0.00001) < 0.001:  # Es forex
                # Cálculo para pares de forex
                contract_size = symbol_info.get('trade_contract_size', 100000)
                point = symbol_info.get('point', 0.00001)
                
                if risk_in_points > 0:
                    # Valor de pip = contract_size * point
                    pip_value = contract_size * point
                    position_size = risk_amount / (risk_in_points * pip_value)
                else:
                    position_size = 0.0
            else:
                # Cálculo para otros instrumentos (como XAUUSD)
                if risk_in_points > 0:
                    position_size = risk_amount / risk_in_points
                else:
                    position_size = 0.0
            
            # Ajustar al volumen mínimo y step
            volume_min = symbol_info.get('volume_min', 0.01)
            volume_step = symbol_info.get('volume_step', 0.01)
            
            if position_size < volume_min:
                position_size = volume_min
            else:
                # Redondear al step más cercano
                position_size = round(position_size / volume_step) * volume_step
            
            # Limitar a máximo razonable
            max_position = min(balance * 0.1, 10.0)  # Max 10% del balance o 10 lotes
            position_size = min(position_size, max_position)
            
            return position_size, risk_amount
            
        except Exception as e:
            logging.error(f"Error calculando tamaño de posición: {e}")
            return 0.01, risk_amount  # Position size mínima como fallback

class EmailNotifier:
    def __init__(self, config: EmailConfig):
        self.config = config
        self.recipient_email = None
    
    def set_recipient(self, email: str):
        """Establecer email destinatario"""
        self.recipient_email = email
    
    def test_connection(self) -> bool:
        """Probar conexión de email"""
        try:
            server = smtplib.SMTP(self.config.smtp_server, self.config.port)
            server.starttls()
            server.login(self.config.email, self.config.password)
            server.quit()
            print("✅ Conexión de email exitosa")
            return True
        except Exception as e:
            print(f"❌ Error de conexión de email: {e}")
            return False
    
    def send_signal_alert(self, signal: TradingSignal) -> bool:
        """Enviar alerta de señal por email"""
        try:
            if not self.recipient_email:
                logging.error("No se ha configurado email destinatario")
                return False
            
            # Crear mensaje
            msg = MIMEMultipart()
            msg['From'] = self.config.email
            msg['To'] = self.recipient_email
            msg['Subject'] = f"🎯 SEÑAL ICT - {signal.pair} {signal.signal.value}"
            
            # Cuerpo del email
            body = self._format_signal_email(signal)
            msg.attach(MIMEText(body, 'html'))
            
            # Enviar email
            server = smtplib.SMTP(self.config.smtp_server, self.config.port)
            server.starttls()
            server.login(self.config.email, self.config.password)
            server.send_message(msg)
            server.quit()
            
            logging.info(f"Email enviado exitosamente para señal {signal.pair}")
            return True
            
        except Exception as e:
            logging.error(f"Error enviando email: {e}")
            return False
    
    def _format_signal_email(self, signal: TradingSignal) -> str:
        """Formatear señal para email HTML"""
        direction_color = "#28a745" if signal.signal == SignalType.BUY else "#dc3545"
        direction_emoji = "🟢" if signal.signal == SignalType.BUY else "🔴"
        
        # Calcular potencial ganancia
        risk_dollars = signal.risk_amount
        if signal.risk_reward > 0:
            potential_profit = risk_dollars * signal.risk_reward
        else:
            potential_profit = 0
        
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 20px;">
            <h2 style="color: {direction_color};">
                {direction_emoji} SEÑAL ICT - {signal.pair}
            </h2>
            
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="margin-top: 0;">📊 Detalles de la Señal</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Par:</td>
                        <td style="padding: 8px;">{signal.pair}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Dirección:</td>
                        <td style="padding: 8px; color: {direction_color};">
                            <strong>{signal.signal.value}</strong>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Entrada:</td>
                        <td style="padding: 8px; font-family: monospace; font-size: 16px;">
                            <strong>{signal.entry_price:.5f}</strong>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Stop Loss:</td>
                        <td style="padding: 8px; font-family: monospace; font-size: 16px;">
                            <strong>{signal.stop_loss:.5f}</strong>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Take Profit:</td>
                        <td style="padding: 8px; font-family: monospace; font-size: 16px;">
                            <strong>{signal.take_profit:.5f}</strong>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Risk/Reward:</td>
                        <td style="padding: 8px;"><strong>{signal.risk_reward:.2f}</strong></td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Confianza:</td>
                        <td style="padding: 8px;"><strong>{signal.confidence*100:.1f}%</strong></td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Tamaño Posición:</td>
                        <td style="padding: 8px;"><strong>{signal.position_size:.2f} lotes</strong></td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Riesgo ($):</td>
                        <td style="padding: 8px; color: #dc3545;"><strong>${signal.risk_amount:.2f}</strong></td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Ganancia Potencial ($):</td>
                        <td style="padding: 8px; color: #28a745;"><strong>${potential_profit:.2f}</strong></td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Estrategia:</td>
                        <td style="padding: 8px;">{signal.strategy}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold;">Timestamp:</td>
                        <td style="padding: 8px;">{signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                    </tr>
                </table>
            </div>
            
            <div style="background-color: #e8f5e8; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745;">
                <h4 style="margin-top: 0; color: #155724;">💡 Instrucciones de Ejecución:</h4>
                <ol style="margin-bottom: 0; color: #155724;">
                    <li>Verificar que el precio esté cerca de la entrada ({signal.entry_price:.5f})</li>
                    <li>Colocar orden {signal.signal.value} con volumen de {signal.position_size:.2f} lotes</li>
                    <li>Establecer Stop Loss en {signal.stop_loss:.5f}</li>
                    <li>Establecer Take Profit en {signal.take_profit:.5f}</li>
                    <li>Confirmar que el riesgo no exceda ${signal.risk_amount:.2f}</li>
                </ol>
            </div>
            
            <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; margin-top: 15px;">
                <h4 style="margin-top: 0; color: #856404;">⚠️ Advertencias Importantes:</h4>
                <ul style="margin-bottom: 0; color: #856404;">
                    <li>Esta señal está optimizada para cuentas de fondeo con spreads altos</li>
                    <li>El tamaño de posición está calculado para riesgo del 1%</li>
                    <li>Confirme las condiciones de mercado antes de ejecutar</li>
                    <li>La señal es válida solo durante las Kill Zones de ICT</li>
                    <li>Siempre verifique el spread actual antes de entrar</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html_body

class ICTTradingSystemEnhanced:
    def __init__(self, config: Dict = None):
        # Cargar configuración
        self.config = config or ConfigManager.load_config()
        
        # Configuraciones originales del sistema ICT
        self.pair_configs = {
            'EURUSD': {
                'spread': 0.00015,  # 1.5 pips
                'min_rr': 2.0,
                'volatility_factor': 1.2,
                'session_bias': 'NY_LONDON'
            },
            'XAUUSD': {
                'spread': 0.35,     # $0.35
                'min_rr': 2.5,
                'volatility_factor': 1.5,
                'session_bias': 'NY'
            },
            'GBPUSD': {
                'spread': 0.00020,  # 2.0 pips
                'min_rr': 2.0,
                'volatility_factor': 1.3,
                'session_bias': 'LONDON'
            },
            'USDJPY': {
                'spread': 0.015,    # 1.5 pips (para JPY)
                'min_rr': 2.0,
                'volatility_factor': 1.1,
                'session_bias': 'ASIA_NY'
            }
        }
        
        # Horarios ICT (UTC)
        self.ict_sessions = {
            'LONDON_OPEN': time(8, 30),
            'LONDON_CLOSE': time(16, 30),
            'NY_OPEN': time(13, 30),
            'NY_CLOSE': time(22, 0),
            'ASIA_KILL_ZONE': time(1, 0),
            'LONDON_KILL_ZONE': time(6, 0),
            'NY_KILL_ZONE': time(13, 30)
        }
        
        # Inicializar componentes
        self.mt5_connection = MetaTraderConnection()
        self.position_calculator = PositionSizeCalculator(
            risk_percentage=self.config.get('trading', {}).get('risk_percentage', 1.0)
        )
        
        # Configurar email si está habilitado
        self.email_notifier = None
        if self.config.get('email', {}).get('enabled', False):
            email_config = EmailConfig(
                smtp_server=self.config['email']['smtp_server'],
                port=self.config['email']['port'],
                email=self.config['email']['email'],
                password=self.config['email']['password']
            )
            self.email_notifier = EmailNotifier(email_config)
            if self.config['email']['recipient']:
                self.email_notifier.set_recipient(self.config['email']['recipient'])
        
        self.indicators = TechnicalIndicators()
        
        # Estado del sistema
        self.balance = self.config.get('trading', {}).get('default_balance', 10000)
        self.last_signals = {}  # Para evitar señales duplicadas
    
    def setup_system(self):
        """Configuración inicial del sistema"""
        print("🚀 Configuración del Sistema ICT Trading")
        print("=" * 50)
        
        # Mostrar configuración cargada
        print(f"📁 Configuración cargada desde config.json:")
        print(f"   📧 Email: {'✅ Habilitado' if self.config.get('email', {}).get('enabled') else '❌ Deshabilitado'}")
        print(f"   🔗 MT5: {'✅ Auto-conectar' if self.config.get('mt5', {}).get('auto_connect') else '❌ Manual'}")
        print(f"   💰 Balance por defecto: ${self.config.get('trading', {}).get('default_balance', 10000):,.2f}")
        
        # Solicitar balance si no se especifica usar MT5
        use_mt5_balance = False
        if MT5_AVAILABLE and self.config.get('mt5', {}).get('auto_connect', True):
            print("\n🔗 Conectando a MetaTrader 5...")
            mt5_config = self.config.get('mt5', {})
            
            if self.mt5_connection.connect(
                login=mt5_config.get('login'),
                password=mt5_config.get('password'),
                server=mt5_config.get('server')
            ):
                # Actualizar balance desde MT5 si está disponible
                mt5_balance = self.mt5_connection.get_balance()
                if mt5_balance > 0:
                    use_mt5_balance = input(f"\n¿Usar balance de MT5 (${mt5_balance:,.2f}) o manual? (mt5/manual): ").lower()
                    if use_mt5_balance in ['mt5', 'm', 'y', 'yes']:
                        self.balance = mt5_balance
                        print(f"✅ Balance actualizado desde MT5: ${self.balance:,.2f}")
                    else:
                        use_mt5_balance = False
            else:
                print("⚠️ No se pudo conectar a MetaTrader 5")
                print("💡 Verifique que MT5 esté abierto y conectado a su cuenta")
        
        # Solicitar balance manual si no se usa MT5
        if not use_mt5_balance:
            try:
                balance_input = input(f"\n💰 Ingrese el balance de su cuenta (actual: ${self.balance:,.2f}): $")
                if balance_input.strip():
                    self.balance = float(balance_input)
                print(f"✅ Balance configurado: ${self.balance:,.2f}")
            except ValueError:
                print("⚠️ Valor inválido, usando balance por defecto")
        
        # Probar conexión de email si está habilitado
        if self.email_notifier:
            print(f"\n📧 Probando conexión de email a {self.config['email']['recipient']}...")
            if self.email_notifier.test_connection():
                print("✅ Email configurado correctamente")
            else:
                print("❌ Error en configuración de email")
                print("💡 Revise config.json y asegúrese de usar contraseña de aplicación para Gmail")
        
        print(f"\n✅ Sistema configurado exitosamente!")
        print(f"💰 Balance: ${self.balance:,.2f}")
        print(f"📧 Email: {self.config.get('email', {}).get('recipient', 'No configurado')}")
        print(f"🔗 MT5: {'Conectado' if self.mt5_connection.connected else 'Desconectado'}")
        
        return True
    
    def get_live_data(self, symbol: str) -> pd.DataFrame:
        """Obtener datos en vivo desde MetaTrader o simulados"""
        if self.mt5_connection.connected:
            # Datos reales desde MT5
            if hasattr(mt5, 'TIMEFRAME_M5'):
                timeframe = mt5.TIMEFRAME_M5
            else:
                timeframe = 5  # Fallback
            
            df = self.mt5_connection.get_symbol_data(symbol, timeframe, 1000)
            if df is not None and len(df) > 0:
                print(f"✅ Datos reales obtenidos de MT5 para {symbol}")
                return df
            else:
                print(f"⚠️ No se pudieron obtener datos de MT5 para {symbol}, usando simulados")
        
        # Datos simulados como fallback
        print(f"📊 Generando datos simulados para {symbol}")
        return self._generate_simulated_data(symbol)
    
    def _generate_simulated_data(self, symbol: str) -> pd.DataFrame:
        """Generar datos simulados para testing"""
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
        np.random.seed(42)
        
        if symbol == 'EURUSD':
            base_price = 1.0950
        elif symbol == 'XAUUSD':
            base_price = 2050.00
        elif symbol == 'GBPUSD':
            base_price = 1.2650
        elif symbol == 'USDJPY':
            base_price = 150.00
        else:
            base_price = 1.0000
        
        # Generar datos más realistas con tendencia
        trend = np.linspace(0, 0.02 * base_price, 1000)  # Tendencia alcista ligera
        noise = np.random.normal(0, base_price * 0.005, 1000)  # Ruido
        
        data = {
            'timestamp': dates,
            'open': base_price + trend + noise,
            'high': base_price + trend + noise + np.abs(np.random.normal(0, base_price * 0.003, 1000)),
            'low': base_price + trend + noise - np.abs(np.random.normal(0, base_price * 0.003, 1000)),
            'close': base_price + trend + noise + np.random.normal(0, base_price * 0.002, 1000),
            'volume': np.random.exponential(1000, 1000)
        }
        
        # Asegurar coherencia OHLC
        for i in range(1000):
            prices = [data['open'][i], data['close'][i]]
            data['high'][i] = max(max(prices), data['high'][i])
            data['low'][i] = min(min(prices), data['low'][i])
        
        return pd.DataFrame(data)
    
    def calculate_position_sizes(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Calcular tamaños de posición para todas las señales"""
        enhanced_signals = []
        
        for signal in signals:
            try:
                # Obtener información del símbolo
                symbol_info = {
                    'trade_contract_size': 100000, 
                    'point': 0.00001, 
                    'volume_min': 0.01, 
                    'volume_step': 0.01
                }
                
                # Ajustar para diferentes tipos de símbolos
                if 'JPY' in signal.pair:
                    symbol_info['point'] = 0.001
                elif 'XAU' in signal.pair or 'GOLD' in signal.pair:
                    symbol_info['trade_contract_size'] = 100
                    symbol_info['point'] = 0.01
                
                if self.mt5_connection.connected:
                    mt5_symbol_info = self.mt5_connection.get_symbol_info(signal.pair)
                    if mt5_symbol_info:
                        symbol_info = mt5_symbol_info
                        print(f"📊 Usando información real de MT5 para {signal.pair}")
                
                # Calcular tamaño de posición
                position_size, risk_amount = self.position_calculator.calculate_position_size(
                    self.balance, signal.entry_price, signal.stop_loss, symbol_info
                )
                
                # Actualizar señal con información de posición
                signal.position_size = position_size
                signal.risk_amount = risk_amount
                
                enhanced_signals.append(signal)
                
            except Exception as e:
                logging.error(f"Error calculando posición para {signal.pair}: {e}")
                continue
        
        return enhanced_signals
    
    def is_duplicate_signal(self, signal: TradingSignal) -> bool:
        """Verificar si la señal es duplicada (evitar spam)"""
        key = f"{signal.pair}_{signal.signal.value}_{signal.strategy}"
        current_time = datetime.now()
        
        if key in self.last_signals:
            time_diff = current_time - self.last_signals[key]
            if time_diff.total_seconds() < 3600:  # 1 hora
                return True
        
        self.last_signals[key] = current_time
        return False
    
    def process_and_send_signals(self, symbols: List[str]):
        """Procesar símbolos y enviar señales por email"""
        all_signals = []
        
        for symbol in symbols:
            try:
                print(f"\n📊 Analizando {symbol}...")
                
                # Obtener datos
                df = self.get_live_data(symbol)
                if df is None or len(df) < 100:
                    print(f"⚠️ No hay suficientes datos para {symbol}")
                    continue
                
                # Generar señales
                signals = self.generate_signals(df, symbol)
                
                if signals:
                    # Calcular tamaños de posición
                    enhanced_signals = self.calculate_position_sizes(signals)
                    
                    for signal in enhanced_signals:
                        # Verificar duplicados
                        if not self.is_duplicate_signal(signal):
                            all_signals.append(signal)
                            
                            # Enviar email si está configurado
                            if self.email_notifier:
                                success = self.email_notifier.send_signal_alert(signal)
                                if success:
                                    print(f"✅ Email enviado para {signal.pair} {signal.signal.value}")
                                else:
                                    print(f"❌ Error enviando email para {signal.pair}")
                            
                            # Mostrar señal en consola
                            print(f"🎯 NUEVA SEÑAL: {signal.pair} {signal.signal.value}")
                            print(f"   Entrada: {signal.entry_price:.5f}")
                            print(f"   SL: {signal.stop_loss:.5f} | TP: {signal.take_profit:.5f}")
                            print(f"   RR: {signal.risk_reward:.2f} | Posición: {signal.position_size:.2f} lotes")
                            print(f"   Confianza: {signal.confidence*100:.1f}% | Riesgo: ${signal.risk_amount:.2f}")
                        else:
                            print(f"⚠️ Señal duplicada ignorada: {signal.pair} {signal.signal.value}")
                else:
                    print(f"❌ No hay señales para {symbol}")
                
            except Exception as e:
                logging.error(f"Error procesando {symbol}: {e}")
                continue
        
        return all_signals
    
    def run_continuous_monitoring(self, symbols: List[str], interval_minutes: int = 5):
        """Ejecutar monitoreo continuo"""
        print(f"\n🚀 Iniciando monitoreo continuo...")
        print(f"📊 Símbolos: {', '.join(symbols)}")
        print(f"⏱️ Intervalo: {interval_minutes} minutos")
        print(f"💰 Balance: ${self.balance:,.2f}")
        print(f"📧 Email: {'✅' if self.email_notifier else '❌'}")
        print(f"🔗 MT5: {'✅' if self.mt5_connection.connected else '❌'}")
        print("Presione Ctrl+C para detener...\n")
        
        # Programar ejecución
        schedule.every(interval_minutes).minutes.do(
            self.process_and_send_signals, symbols
        )
        
        # Ejecutar una vez inmediatamente
        print("🔍 Ejecutando análisis inicial...")
        self.process_and_send_signals(symbols)
        
        try:
            while True:
                schedule.run_pending()
                time_module.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoreo detenido por el usuario")
        finally:
            if self.mt5_connection.connected:
                self.mt5_connection.disconnect()

    # =================== MÉTODOS ORIGINALES ICT ===================
    
    def calculate_market_structure(self, df: pd.DataFrame) -> MarketStructure:
        """Determina la estructura del mercado basada en SMC (Smart Money Concepts)"""
        high_prices = df['high'].rolling(window=20).max()
        low_prices = df['low'].rolling(window=20).min()
        
        recent_highs = df['high'].tail(10).max()
        recent_lows = df['low'].tail(10).min()
        
        # Lógica BOS (Break of Structure)
        if len(high_prices) > 21 and recent_highs > high_prices.iloc[-21:-11].max():
            return MarketStructure.BULLISH
        elif len(low_prices) > 21 and recent_lows < low_prices.iloc[-21:-11].min():
            return MarketStructure.BEARISH
        else:
            return MarketStructure.RANGING

    def identify_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Identifica Order Blocks usando volumen y estructura de velas"""
        order_blocks = []
        
        for i in range(20, len(df)-5):
            candle = df.iloc[i]
            volume_avg = df['volume'].rolling(10).mean().iloc[i]
            
            # Condiciones para Order Block bajista
            if (candle['close'] < candle['open'] and  # Vela bajista
                candle['volume'] > volume_avg * 1.5 and  # Alto volumen
                abs(candle['close'] - candle['open']) > (candle['high'] - candle['low']) * 0.6):  # Cuerpo grande
                
                # Verificar si hay mitigación posterior
                future_prices = df['low'].iloc[i+1:i+6]
                if len(future_prices) > 0 and future_prices.min() >= candle['low']:
                    order_blocks.append({
                        'type': 'bearish',
                        'high': candle['high'],
                        'low': candle['low'],
                        'index': i,
                        'mitigated': False
                    })
            
            # Condiciones para Order Block alcista
            elif (candle['close'] > candle['open'] and
                  candle['volume'] > volume_avg * 1.5 and
                  abs(candle['close'] - candle['open']) > (candle['high'] - candle['low']) * 0.6):
                
                future_prices = df['high'].iloc[i+1:i+6]
                if len(future_prices) > 0 and future_prices.max() <= candle['high']:
                    order_blocks.append({
                        'type': 'bullish',
                        'high': candle['high'],
                        'low': candle['low'],
                        'index': i,
                        'mitigated': False
                    })
        
        return order_blocks

    def calculate_footprint_analysis(self, df: pd.DataFrame) -> Dict:
        """Análisis de footprint simplificado basado en volumen y precio"""
        
        # Volume Profile aproximado
        price_levels = np.linspace(df['low'].min(), df['high'].max(), 50)
        volume_profile = np.zeros(len(price_levels))
        
        for i, row in df.iterrows():
            # Distribuir volumen en niveles de precio
            candle_range = row['high'] - row['low']
            if candle_range > 0:
                levels_in_range = (price_levels >= row['low']) & (price_levels <= row['high'])
                if np.sum(levels_in_range) > 0:
                    volume_per_level = row['volume'] / np.sum(levels_in_range)
                    volume_profile[levels_in_range] += volume_per_level
        
        # Identificar POC (Point of Control)
        poc_index = np.argmax(volume_profile)
        poc_price = price_levels[poc_index]
        
        # Value Area (70% del volumen)
        total_volume = np.sum(volume_profile)
        target_volume = total_volume * 0.7
        
        sorted_indices = np.argsort(volume_profile)[::-1]
        cumulative_volume = 0
        value_area_indices = []
        
        for idx in sorted_indices:
            cumulative_volume += volume_profile[idx]
            value_area_indices.append(idx)
            if cumulative_volume >= target_volume:
                break
        
        va_high = price_levels[max(value_area_indices)] if value_area_indices else poc_price
        va_low = price_levels[min(value_area_indices)] if value_area_indices else poc_price
        
        return {
            'poc': poc_price,
            'value_area_high': va_high,
            'value_area_low': va_low,
            'volume_profile': volume_profile,
            'price_levels': price_levels
        }

    def calculate_ict_indicators(self, df: pd.DataFrame) -> Dict:
        """Calcula indicadores específicos de ICT"""
        
        # Fair Value Gaps (FVG)
        fvg_bullish = []
        fvg_bearish = []
        
        for i in range(2, len(df)):
            prev_candle = df.iloc[i-2]
            current_candle = df.iloc[i-1]
            next_candle = df.iloc[i]
            
            # FVG Alcista
            if (prev_candle['high'] < next_candle['low'] and
                current_candle['close'] > current_candle['open']):
                fvg_bullish.append({
                    'top': next_candle['low'],
                    'bottom': prev_candle['high'],
                    'index': i
                })
            
            # FVG Bajista
            if (prev_candle['low'] > next_candle['high'] and
                current_candle['close'] < current_candle['open']):
                fvg_bearish.append({
                    'top': prev_candle['low'],
                    'bottom': next_candle['high'],
                    'index': i
                })
        
        # Liquidity Levels (Swing Highs/Lows)
        swing_highs = []
        swing_lows = []
        
        for i in range(10, len(df)-10):
            window = df.iloc[i-5:i+6]
            current_price = df.iloc[i]
            
            if current_price['high'] == window['high'].max():
                swing_highs.append({'price': current_price['high'], 'index': i})
            
            if current_price['low'] == window['low'].min():
                swing_lows.append({'price': current_price['low'], 'index': i})
        
        return {
            'fvg_bullish': fvg_bullish,
            'fvg_bearish': fvg_bearish,
            'swing_highs': swing_highs,
            'swing_lows': swing_lows
        }

    def is_kill_zone_active(self, current_time: time) -> bool:
        """Verifica si estamos en una Kill Zone de ICT"""
        kill_zones = [
            (time(1, 0), time(4, 0)),    # Asia Kill Zone
            (time(6, 0), time(9, 0)),    # London Kill Zone
            (time(13, 30), time(16, 30)) # NY Kill Zone
        ]
        
        for start, end in kill_zones:
            if start <= current_time <= end:
                return True
        return False

    def calculate_spread_adjusted_levels(self, entry: float, sl: float, tp: float, 
                                       pair: str, signal_type: SignalType) -> Tuple[float, float, float]:
        """Ajusta niveles considerando el spread alto de cuentas de fondeo"""
        spread = self.pair_configs.get(pair, {}).get('spread', 0.00020)
        min_rr = self.pair_configs.get(pair, {}).get('min_rr', 2.0)
        
        if signal_type == SignalType.BUY:
            # Para compras, el entry se ajusta al ask
            entry_adj = entry + spread
            sl_adj = sl
            tp_adj = tp
            
            # Verificar que mantenemos RR mínimo después del spread
            risk = entry_adj - sl_adj
            reward = tp_adj - entry_adj
            
            if risk > 0 and reward / risk >= min_rr:
                return entry_adj, sl_adj, tp_adj
            else:
                # Ajustar TP para mantener RR
                tp_adj = entry_adj + (risk * min_rr)
                return entry_adj, sl_adj, tp_adj
        
        else:  # SELL
            # Para ventas, el entry se ajusta al bid
            entry_adj = entry - spread
            sl_adj = sl
            tp_adj = tp
            
            risk = sl_adj - entry_adj
            reward = entry_adj - tp_adj
            
            if risk > 0 and reward / risk >= min_rr:
                return entry_adj, sl_adj, tp_adj
            else:
                tp_adj = entry_adj - (risk * min_rr)
                return entry_adj, sl_adj, tp_adj

    def generate_signals(self, df: pd.DataFrame, pair: str) -> List[TradingSignal]:
        """Generador principal de señales"""
        signals = []
        
        try:
            # Calcular indicadores
            market_structure = self.calculate_market_structure(df)
            order_blocks = self.identify_order_blocks(df)
            footprint = self.calculate_footprint_analysis(df)
            ict_data = self.calculate_ict_indicators(df)
            
            # Indicadores técnicos usando nuestra implementación
            df['ema_20'] = self.indicators.ema(df['close'], 20)
            df['ema_50'] = self.indicators.ema(df['close'], 50)
            df['rsi'] = self.indicators.rsi(df['close'], 14)
            df['atr'] = self.indicators.atr(df['high'], df['low'], df['close'], 14)
            
            current_time = datetime.now().time()
            
            # Solo generar señales en Kill Zones (comentado para testing)
            # if not self.is_kill_zone_active(current_time):
            #     return signals
            
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            if pd.isna(atr) or atr <= 0:
                atr = abs(df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()) * 0.1
            
            # Estrategia 1: Order Block + FVG
            for ob in order_blocks[-5:]:  # Solo los últimos 5 order blocks
                if not ob['mitigated'] and len(df) - ob['index'] <= 20:  # Order block reciente
                    
                    # Buscar FVG cercano
                    relevant_fvg = None
                    if ob['type'] == 'bullish':
                        for fvg in ict_data['fvg_bullish'][-10:]:  # Últimos 10 FVG
                            if abs(fvg['top'] - ob['high']) < atr * 0.5:
                                relevant_fvg = fvg
                                break
                    else:
                        for fvg in ict_data['fvg_bearish'][-10:]:
                            if abs(fvg['bottom'] - ob['low']) < atr * 0.5:
                                relevant_fvg = fvg
                                break
                    
                    if relevant_fvg and ob['type'] == 'bullish' and market_structure == MarketStructure.BULLISH:
                        entry = ob['low']
                        sl = entry - atr * 1.5
                        tp = entry + atr * 3.0
                        
                        if sl > 0 and tp > entry:  # Validar niveles
                            entry_adj, sl_adj, tp_adj = self.calculate_spread_adjusted_levels(
                                entry, sl, tp, pair, SignalType.BUY)
                            
                            confidence = 0.75
                            if footprint['value_area_low'] <= current_price <= footprint['value_area_high']:
                                confidence += 0.15  # Precio en Value Area
                            
                            rr = (tp_adj - entry_adj) / (entry_adj - sl_adj) if (entry_adj - sl_adj) > 0 else 0
                            
                            if rr >= 1.5:  # Mínimo RR aceptable
                                signals.append(TradingSignal(
                                    timestamp=datetime.now(),
                                    pair=pair,
                                    signal=SignalType.BUY,
                                    entry_price=entry_adj,
                                    stop_loss=sl_adj,
                                    take_profit=tp_adj,
                                    risk_reward=rr,
                                    confidence=confidence,
                                    strategy="Order Block + FVG Bullish",
                                    spread_adjusted=True
                                ))
                    
                    elif relevant_fvg and ob['type'] == 'bearish' and market_structure == MarketStructure.BEARISH:
                        entry = ob['high']
                        sl = entry + atr * 1.5
                        tp = entry - atr * 3.0
                        
                        if tp > 0 and sl > entry:  # Validar niveles
                            entry_adj, sl_adj, tp_adj = self.calculate_spread_adjusted_levels(
                                entry, sl, tp, pair, SignalType.SELL)
                            
                            confidence = 0.75
                            if footprint['value_area_low'] <= current_price <= footprint['value_area_high']:
                                confidence += 0.15
                            
                            rr = (entry_adj - tp_adj) / (sl_adj - entry_adj) if (sl_adj - entry_adj) > 0 else 0
                            
                            if rr >= 1.5:
                                signals.append(TradingSignal(
                                    timestamp=datetime.now(),
                                    pair=pair,
                                    signal=SignalType.SELL,
                                    entry_price=entry_adj,
                                    stop_loss=sl_adj,
                                    take_profit=tp_adj,
                                    risk_reward=rr,
                                    confidence=confidence,
                                    strategy="Order Block + FVG Bearish",
                                    spread_adjusted=True
                                ))
            
            # Estrategia 2: Liquidity Sweep + Reversal
            if len(ict_data['swing_highs']) > 0 and len(ict_data['swing_lows']) > 0:
                recent_high = max(ict_data['swing_highs'], key=lambda x: x['index'])
                recent_low = max(ict_data['swing_lows'], key=lambda x: x['index'])  # Cambiado a max para obtener el más reciente
                
                rsi_current = df['rsi'].iloc[-1]
                if pd.isna(rsi_current):
                    rsi_current = 50  # Valor neutral si RSI no está disponible
                
                # Sweep de liquidez en high con reversión bajista
                if (current_price > recent_high['price'] + atr * 0.2 and 
                    rsi_current > 70 and 
                    market_structure in [MarketStructure.BEARISH, MarketStructure.RANGING]):
                    
                    entry = current_price
                    sl = recent_high['price'] + atr * 0.5
                    tp = footprint['poc']
                    
                    if entry > tp and sl > entry and tp > 0:  # Validar setup
                        entry_adj, sl_adj, tp_adj = self.calculate_spread_adjusted_levels(
                            entry, sl, tp, pair, SignalType.SELL)
                        
                        rr = (entry_adj - tp_adj) / (sl_adj - entry_adj) if (sl_adj - entry_adj) > 0 else 0
                        
                        if rr >= 1.5:
                            signals.append(TradingSignal(
                                timestamp=datetime.now(),
                                pair=pair,
                                signal=SignalType.SELL,
                                entry_price=entry_adj,
                                stop_loss=sl_adj,
                                take_profit=tp_adj,
                                risk_reward=rr,
                                confidence=0.70,
                                strategy="Liquidity Sweep Reversal Bearish",
                                spread_adjusted=True
                            ))
                
                # Sweep de liquidez en low con reversión alcista
                elif (current_price < recent_low['price'] - atr * 0.2 and 
                      rsi_current < 30 and 
                      market_structure in [MarketStructure.BULLISH, MarketStructure.RANGING]):
                    
                    entry = current_price
                    sl = recent_low['price'] - atr * 0.5
                    tp = footprint['poc']
                    
                    if entry < tp and sl < entry and sl > 0:  # Validar setup
                        entry_adj, sl_adj, tp_adj = self.calculate_spread_adjusted_levels(
                            entry, sl, tp, pair, SignalType.BUY)
                        
                        rr = (tp_adj - entry_adj) / (entry_adj - sl_adj) if (entry_adj - sl_adj) > 0 else 0
                        
                        if rr >= 1.5:
                            signals.append(TradingSignal(
                                timestamp=datetime.now(),
                                pair=pair,
                                signal=SignalType.BUY,
                                entry_price=entry_adj,
                                stop_loss=sl_adj,
                                take_profit=tp_adj,
                                risk_reward=rr,
                                confidence=0.70,
                                strategy="Liquidity Sweep Reversal Bullish",
                                spread_adjusted=True
                            ))
            
            # Estrategia 3: EMA Crossover + Structure (Nueva estrategia)
            if len(df) > 50:
                ema_20_current = df['ema_20'].iloc[-1]
                ema_50_current = df['ema_50'].iloc[-1]
                ema_20_prev = df['ema_20'].iloc[-2]
                ema_50_prev = df['ema_50'].iloc[-2]
                
                # Crossover alcista
                if (ema_20_prev <= ema_50_prev and ema_20_current > ema_50_current and 
                    market_structure == MarketStructure.BULLISH and
                    not pd.isna(ema_20_current) and not pd.isna(ema_50_current)):
                    
                    entry = current_price
                    sl = ema_50_current - atr * 0.5
                    tp = entry + atr * 2.5
                    
                    if sl > 0 and tp > entry:
                        entry_adj, sl_adj, tp_adj = self.calculate_spread_adjusted_levels(
                            entry, sl, tp, pair, SignalType.BUY)
                        
                        rr = (tp_adj - entry_adj) / (entry_adj - sl_adj) if (entry_adj - sl_adj) > 0 else 0
                        
                        if rr >= 1.5:
                            signals.append(TradingSignal(
                                timestamp=datetime.now(),
                                pair=pair,
                                signal=SignalType.BUY,
                                entry_price=entry_adj,
                                stop_loss=sl_adj,
                                take_profit=tp_adj,
                                risk_reward=rr,
                                confidence=0.65,
                                strategy="EMA Crossover Bullish",
                                spread_adjusted=True
                            ))
                
                # Crossover bajista
                elif (ema_20_prev >= ema_50_prev and ema_20_current < ema_50_current and 
                      market_structure == MarketStructure.BEARISH and
                      not pd.isna(ema_20_current) and not pd.isna(ema_50_current)):
                    
                    entry = current_price
                    sl = ema_50_current + atr * 0.5
                    tp = entry - atr * 2.5
                    
                    if tp > 0 and sl > entry:
                        entry_adj, sl_adj, tp_adj = self.calculate_spread_adjusted_levels(
                            entry, sl, tp, pair, SignalType.SELL)
                        
                        rr = (entry_adj - tp_adj) / (sl_adj - entry_adj) if (sl_adj - entry_adj) > 0 else 0
                        
                        if rr >= 1.5:
                            signals.append(TradingSignal(
                                timestamp=datetime.now(),
                                pair=pair,
                                signal=SignalType.SELL,
                                entry_price=entry_adj,
                                stop_loss=sl_adj,
                                take_profit=tp_adj,
                                risk_reward=rr,
                                confidence=0.65,
                                strategy="EMA Crossover Bearish",
                                spread_adjusted=True
                            ))
            
        except Exception as e:
            logging.error(f"Error generando señales para {pair}: {e}")
            return []
        
        # Filtrar señales por calidad
        quality_signals = []
        min_rr = self.pair_configs.get(pair, {}).get('min_rr', 2.0)
        
        for signal in signals:
            if (signal.confidence >= 0.60 and 
                signal.risk_reward >= 1.5 and  # RR mínimo más flexible
                signal.entry_price > 0 and
                signal.stop_loss > 0 and
                signal.take_profit > 0):
                quality_signals.append(signal)
        
        return quality_signals

# =================== CONFIGURACIÓN Y FUNCIONES PRINCIPALES ===================

def create_sample_config():
    """Crear archivo de configuración de ejemplo"""
    sample_config = {
        "email": {
            "enabled": True,
            "smtp_server": "smtp.gmail.com",
            "port": 587,
            "email": "tu_email@gmail.com",
            "password": "tu_contraseña_de_aplicacion",
            "recipient": "destinatario@gmail.com"
        },
        "mt5": {
            "auto_connect": True,
            "login": None,
            "password": "",
            "server": ""
        },
        "trading": {
            "default_balance": 10000,
            "risk_percentage": 1.0,
            "default_symbols": ["EURUSD", "XAUUSD", "GBPUSD", "USDJPY"],
            "default_interval": 5
        }
    }
    
    with open('config_sample.json', 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=4, ensure_ascii=False)
    
    print("📁 Archivo config_sample.json creado como ejemplo")

def setup_email_config():
    """Configurar email interactivamente y guardar en config.json"""
    config = ConfigManager.load_config()
    
    print("\n📧 Configuración de Email")
    print("=" * 30)
    
    # Configuraciones comunes de SMTP
    smtp_configs = {
        "1": {"name": "Gmail", "server": "smtp.gmail.com", "port": 587},
        "2": {"name": "Outlook/Hotmail", "server": "smtp-mail.outlook.com", "port": 587},
        "3": {"name": "Yahoo", "server": "smtp.mail.yahoo.com", "port": 587},
        "4": {"name": "Personalizado", "server": "", "port": 587}
    }
    
    print("Seleccione su proveedor de email:")
    for key, smtp_config in smtp_configs.items():
        print(f"{key}. {smtp_config['name']}")
    
    choice = input("\nElija una opción (1-4): ")
    
    if choice in smtp_configs:
        smtp_config = smtp_configs[choice]
        
        if choice == "4":  # Personalizado
            server = input("Servidor SMTP: ")
            port = int(input("Puerto SMTP (587): ") or "587")
        else:
            server = smtp_config["server"]
            port = smtp_config["port"]
        
        email = input("Su email: ")
        password = input("Contraseña de aplicación: ")
        recipient = input("Email destinatario: ")
        
        # Actualizar configuración
        config["email"] = {
            "enabled": True,
            "smtp_server": server,
            "port": port,
            "email": email,
            "password": password,
            "recipient": recipient
        }
        
        # Guardar configuración
        ConfigManager.save_config(config)
        
        # Probar conexión
        email_config = EmailConfig(server, port, email, password)
        notifier = EmailNotifier(email_config)
        notifier.set_recipient(recipient)
        
        if notifier.test_connection():
            print("✅ Configuración de email guardada y probada exitosamente")
        else:
            print("⚠️ Configuración guardada pero hay problemas de conexión")
    else:
        print("❌ Opción inválida")

def main():
    """Función principal del sistema"""
    print("🎯 ICT Trading System Enhanced con Configuración JSON")
    print("=" * 60)
    
    # Verificar si existe configuración
    if not os.path.exists('config.json'):
        print("📁 No se encontró config.json")
        setup = input("¿Desea configurar el sistema ahora? (y/n): ")
        if setup.lower() == 'y':
            setup_email_config()
        else:
            print("💡 Se creará configuración por defecto...")
    
    # Inicializar sistema
    trading_system = ICTTradingSystemEnhanced()
    
    # Configurar sistema
    if not trading_system.setup_system():
        print("❌ Error en configuración del sistema")
        return
    
    # Seleccionar símbolos a monitorear
    print("\n📊 Símbolos disponibles:")
    available_symbols = list(trading_system.pair_configs.keys())
    default_symbols = trading_system.config.get('trading', {}).get('default_symbols', available_symbols)
    
    for i, symbol in enumerate(available_symbols, 1):
        default_mark = " (por defecto)" if symbol in default_symbols else ""
        print(f"{i}. {symbol}{default_mark}")
    
    symbol_choice = input(f"\nSeleccione símbolos (ej: 1,2,4 o 'all' o Enter para usar por defecto): ")
    
    if not symbol_choice.strip():
        selected_symbols = default_symbols
    elif symbol_choice.lower() == 'all':
        selected_symbols = available_symbols
    else:
        try:
            indices = [int(x.strip()) - 1 for x in symbol_choice.split(',')]
            selected_symbols = [available_symbols[i] for i in indices if 0 <= i < len(available_symbols)]
        except:
            selected_symbols = default_symbols
    
    print(f"✅ Símbolos seleccionados: {', '.join(selected_symbols)}")
    
    # Seleccionar modo de ejecución
    print("\n🚀 Modo de ejecución:")
    print("1. Análisis único")
    print("2. Monitoreo continuo")
    print("3. Configurar email")
    print("4. Crear config de ejemplo")
    
    mode = input("Seleccione modo (1-4): ")
    
    if mode == "1":
        # Análisis único
        print("\n📊 Ejecutando análisis único...")
        signals = trading_system.process_and_send_signals(selected_symbols)
        
        if signals:
            print(f"\n✅ Se encontraron {len(signals)} señales:")
            for signal in signals:
                print(f"📈 {signal.pair} {signal.signal.value} - RR: {signal.risk_reward:.2f} - Confianza: {signal.confidence*100:.1f}%")
        else:
            print("❌ No se encontraron señales en este momento")
    
    elif mode == "2":
        # Monitoreo continuo
        default_interval = trading_system.config.get('trading', {}).get('default_interval', 5)
        interval = input(f"\nIntervalo de monitoreo en minutos (default: {default_interval}): ")
        try:
            interval_minutes = int(interval) if interval else default_interval
        except:
            interval_minutes = default_interval
        
        trading_system.run_continuous_monitoring(selected_symbols, interval_minutes)
    
    elif mode == "3":
        # Configurar email
        setup_email_config()
        print("✅ Configuración de email completada. Reinicie el programa para aplicar cambios.")
    
    elif mode == "4":
        # Crear config de ejemplo
        create_sample_config()
        print("✅ Archivo config_sample.json creado como referencia")
    
    else:
        print("❌ Opción inválida")

def quick_test():
    """Test rápido del sistema sin configuración completa"""
    print("🧪 Modo Test Rápido")
    
    # Sistema con configuración mínima
    test_config = {
        "email": {"enabled": False},
        "mt5": {"auto_connect": False},
        "trading": {"default_balance": 10000, "risk_percentage": 1.0}
    }
    
    system = ICTTradingSystemEnhanced(test_config)
    system.balance = 10000.0
    
    # Test con todos los símbolos
    test_symbols = ['EURUSD', 'XAUUSD', 'GBPUSD']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")
        df = system._generate_simulated_data(symbol)
        signals = system.generate_signals(df, symbol)
        
        if signals:
            enhanced_signals = system.calculate_position_sizes(signals)
            
            print(f"✅ {len(enhanced_signals)} señales para {symbol}:")
            for signal in enhanced_signals:
                print(f"   {signal.signal.value} | RR: {signal.risk_reward:.2f} | "
                      f"Posición: {signal.position_size:.2f} lotes | "
                      f"Estrategia: {signal.strategy}")
        else:
            print(f"❌ No hay señales para {symbol}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "config":
        setup_email_config()
    else:
        main()