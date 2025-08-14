import pandas as pd
import numpy as np
from datetime import datetime, time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import json
import os
import schedule
import time as time_module
import logging
import subprocess
import psutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ict_trading.log'),
        logging.StreamHandler()
    ]
)

# Intentar importar MetaTrader5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    logging.info("✅ MetaTrader5 disponible")
except ImportError:
    MT5_AVAILABLE = False
    logging.warning("⚠️ MetaTrader5 no disponible - usando modo simulado")

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

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
    confluences: List[str]
    position_size: float = 0.0
    risk_amount: float = 0.0
    chart_image: bytes = None

class ConfigManager:
    """Gestor de configuración mejorado con validación completa"""
    
    @staticmethod
    def create_default_config():
        """Crear configuración por defecto con todos los parámetros necesarios"""
        default_config = {
            "email": {
                "enabled": True,
                "smtp_server": "smtp.gmail.com",
                "port": 587,
                "email": "",
                "password": "",
                "recipient": ""
            },
            "mt5": {
                "auto_connect": True,
                "auto_open": True,  # Abrir MT5 automáticamente si no está abierto
                "terminal_path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
                "login": None,
                "password": "",
                "server": "",
                "retry_attempts": 5,
                "retry_delay": 5  # segundos entre reintentos
            },
            "trading": {
                "default_balance": 10000,
                "risk_percentage": 1.0,
                "default_symbols": ["EURUSD", "XAUUSD", "GBPUSD"],
                "default_interval": 5,
                "min_confidence": 60,  # % mínimo de confianza para generar señal
                "min_risk_reward": 1.5
            },
            "indicators": {
                "ema_fast": 20,
                "ema_slow": 50,
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "atr_period": 14,
                "volume_multiplier": 1.5,
                "swing_lookback": 10
            },
            "confluences": {
                "enabled": {
                    "market_structure": True,
                    "order_blocks": True,
                    "fair_value_gaps": True,
                    "liquidity_sweeps": True,
                    "ema_crossover": True,
                    "rsi_divergence": True,
                    "volume_analysis": True,
                    "kill_zones": True
                },
                "weights": {
                    "market_structure": 20,
                    "order_blocks": 15,
                    "fair_value_gaps": 15,
                    "liquidity_sweeps": 10,
                    "ema_crossover": 10,
                    "rsi_divergence": 10,
                    "volume_analysis": 10,
                    "kill_zones": 10
                }
            },
            "pairs": {
                "EURUSD": {
                    "spread": 0.00015,
                    "min_rr": 2.0,
                    "lot_size": 0.01,
                    "volatility_factor": 1.2
                },
                "XAUUSD": {
                    "spread": 0.35,
                    "min_rr": 2.5,
                    "lot_size": 0.01,
                    "volatility_factor": 1.5
                },
                "GBPUSD": {
                    "spread": 0.00020,
                    "min_rr": 2.0,
                    "lot_size": 0.01,
                    "volatility_factor": 1.3
                }
            }
        }
        
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4, ensure_ascii=False)
        
        logging.info("📁 Archivo config.json creado con configuración por defecto")
        return default_config
    
    @staticmethod
    def load_config():
        """Cargar y validar configuración"""
        try:
            if not os.path.exists('config.json'):
                logging.info("📁 No se encontró config.json, creando archivo por defecto...")
                return ConfigManager.create_default_config()
            
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validar estructura mínima
            required_keys = ['email', 'mt5', 'trading', 'indicators', 'confluences']
            for key in required_keys:
                if key not in config:
                    logging.warning(f"⚠️ Falta la sección '{key}' en config.json, regenerando...")
                    return ConfigManager.create_default_config()
            
            logging.info("✅ Configuración cargada y validada desde config.json")
            return config
            
        except Exception as e:
            logging.error(f"Error cargando configuración: {e}")
            return ConfigManager.create_default_config()

class MT5AutoConnector:
    """Clase para manejar la conexión automática con MT5"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.mt5_config = config.get('mt5', {})
        self.connected = False
        self.account_info = None
    
    def is_mt5_running(self) -> bool:
        """Verificar si MT5 está en ejecución"""
        for proc in psutil.process_iter(['name']):
            if 'terminal64.exe' in proc.info['name'].lower():
                return True
        return False
    
    def open_mt5(self) -> bool:
        """Abrir MT5 automáticamente"""
        terminal_path = self.mt5_config.get('terminal_path', '')
        
        # Buscar rutas comunes de MT5
        possible_paths = [
            terminal_path,
            "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
            "C:\\Program Files (x86)\\MetaTrader 5\\terminal64.exe",
            f"C:\\Users\\{os.getlogin()}\\AppData\\Roaming\\MetaQuotes\\Terminal\\terminal64.exe"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    logging.info(f"🚀 Abriendo MT5 desde: {path}")
                    subprocess.Popen([path])
                    time_module.sleep(10)  # Esperar a que MT5 se abra
                    return True
                except Exception as e:
                    logging.error(f"Error abriendo MT5: {e}")
        
        logging.error("❌ No se pudo encontrar MT5 en las rutas conocidas")
        return False
    
    def connect_with_retry(self) -> bool:
        """Conectar a MT5 con reintentos automáticos"""
        if not MT5_AVAILABLE:
            logging.error("MetaTrader5 module no está disponible")
            return False
        
        retry_attempts = self.mt5_config.get('retry_attempts', 5)
        retry_delay = self.mt5_config.get('retry_delay', 5)
        
        for attempt in range(retry_attempts):
            logging.info(f"🔄 Intento de conexión {attempt + 1}/{retry_attempts}")
            
            # Verificar si MT5 está abierto
            if not self.is_mt5_running():
                if self.mt5_config.get('auto_open', True):
                    logging.info("MT5 no está abierto. Intentando abrir...")
                    if not self.open_mt5():
                        logging.warning("No se pudo abrir MT5 automáticamente")
                        continue
                else:
                    logging.warning("MT5 no está abierto. Por favor, ábralo manualmente")
                    time_module.sleep(retry_delay)
                    continue
            
            # Intentar inicializar MT5
            if mt5.initialize():
                logging.info("✅ MT5 inicializado correctamente")
                
                # Intentar login si hay credenciales
                if all([self.mt5_config.get('login'), 
                       self.mt5_config.get('password'), 
                       self.mt5_config.get('server')]):
                    
                    if mt5.login(
                        self.mt5_config['login'],
                        self.mt5_config['password'],
                        self.mt5_config['server']
                    ):
                        logging.info("✅ Login exitoso en MT5")
                    else:
                        logging.warning("⚠️ No se pudo hacer login, usando cuenta por defecto")
                
                # Obtener información de cuenta
                self.account_info = mt5.account_info()
                if self.account_info:
                    self.connected = True
                    logging.info(f"✅ Conectado a cuenta: {self.account_info.login}")
                    logging.info(f"💰 Balance: ${self.account_info.balance:,.2f}")
                    return True
                else:
                    logging.warning("⚠️ No se pudo obtener información de cuenta")
            
            if attempt < retry_attempts - 1:
                logging.info(f"⏳ Esperando {retry_delay} segundos antes de reintentar...")
                time_module.sleep(retry_delay)
        
        logging.error("❌ No se pudo conectar a MT5 después de todos los intentos")
        return False
    
    def get_balance(self) -> float:
        """Obtener balance actual"""
        if self.connected and self.account_info:
            self.account_info = mt5.account_info()
            return self.account_info.balance if self.account_info else 0.0
        return 0.0
    
    def disconnect(self):
        """Desconectar de MT5"""
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            logging.info("Desconectado de MT5")

class TechnicalIndicators:
    """Calculadora de indicadores técnicos"""
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 10) -> str:
        """Detectar divergencias entre precio e indicador"""
        if len(price) < lookback * 2:
            return "none"
        
        # Encontrar máximos y mínimos locales
        price_highs = []
        price_lows = []
        ind_highs = []
        ind_lows = []
        
        for i in range(lookback, len(price) - lookback):
            if price.iloc[i] == price.iloc[i-lookback:i+lookback+1].max():
                price_highs.append(i)
                ind_highs.append(indicator.iloc[i])
            if price.iloc[i] == price.iloc[i-lookback:i+lookback+1].min():
                price_lows.append(i)
                ind_lows.append(indicator.iloc[i])
        
        # Divergencia bajista (precio hace HH, indicador hace LH)
        if len(price_highs) >= 2:
            if (price.iloc[price_highs[-1]] > price.iloc[price_highs[-2]] and 
                ind_highs[-1] < ind_highs[-2]):
                return "bearish"
        
        # Divergencia alcista (precio hace LL, indicador hace HL)
        if len(price_lows) >= 2:
            if (price.iloc[price_lows[-1]] < price.iloc[price_lows[-2]] and 
                ind_lows[-1] > ind_lows[-2]):
                return "bullish"
        
        return "none"

class ChartGenerator:
    """Generador de gráficos para señales"""
    
    @staticmethod
    def create_signal_chart(df: pd.DataFrame, signal: TradingSignal, 
                           indicators_config: Dict) -> bytes:
        """Crear gráfico de la señal con indicadores"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), 
                                                gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Limitar datos a las últimas 100 velas para mejor visualización
            df_plot = df.tail(100).copy()
            
            # Gráfico principal de precios
            ax1.plot(df_plot.index, df_plot['close'], label='Precio', color='blue', linewidth=1)
            
            # EMAs
            if 'ema_fast' in df_plot.columns:
                ax1.plot(df_plot.index, df_plot['ema_fast'], 
                        label=f'EMA {indicators_config["ema_fast"]}', 
                        color='orange', alpha=0.7)
            if 'ema_slow' in df_plot.columns:
                ax1.plot(df_plot.index, df_plot['ema_slow'], 
                        label=f'EMA {indicators_config["ema_slow"]}', 
                        color='purple', alpha=0.7)
            
            # Marcar niveles de entrada, SL y TP
            ax1.axhline(y=signal.entry_price, color='green', linestyle='--', 
                       label=f'Entrada: {signal.entry_price:.5f}')
            ax1.axhline(y=signal.stop_loss, color='red', linestyle='--', 
                       label=f'SL: {signal.stop_loss:.5f}')
            ax1.axhline(y=signal.take_profit, color='blue', linestyle='--', 
                       label=f'TP: {signal.take_profit:.5f}')
            
            # Señal
            last_idx = df_plot.index[-1]
            if signal.signal == SignalType.BUY:
                ax1.scatter(last_idx, signal.entry_price, color='green', 
                          marker='^', s=200, zorder=5)
            else:
                ax1.scatter(last_idx, signal.entry_price, color='red', 
                          marker='v', s=200, zorder=5)
            
            ax1.set_title(f'{signal.pair} - Señal {signal.signal.value} - Confianza: {signal.confidence:.1f}%')
            ax1.set_ylabel('Precio')
            ax1.legend(loc='best', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # RSI
            if 'rsi' in df_plot.columns:
                ax2.plot(df_plot.index, df_plot['rsi'], label='RSI', color='purple')
                ax2.axhline(y=indicators_config['rsi_overbought'], 
                          color='red', linestyle='--', alpha=0.5)
                ax2.axhline(y=indicators_config['rsi_oversold'], 
                          color='green', linestyle='--', alpha=0.5)
                ax2.fill_between(df_plot.index, indicators_config['rsi_oversold'], 
                                indicators_config['rsi_overbought'], alpha=0.1)
                ax2.set_ylabel('RSI')
                ax2.set_ylim(0, 100)
                ax2.grid(True, alpha=0.3)
            
            # Volumen
            colors = ['green' if df_plot['close'].iloc[i] >= df_plot['open'].iloc[i] 
                     else 'red' for i in range(len(df_plot))]
            ax3.bar(df_plot.index, df_plot['volume'], color=colors, alpha=0.5)
            ax3.set_ylabel('Volumen')
            ax3.set_xlabel('Tiempo')
            ax3.grid(True, alpha=0.3)
            
            # Formato de fecha en eje x
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Convertir a bytes
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            chart_bytes = buffer.read()
            plt.close()
            
            return chart_bytes
            
        except Exception as e:
            logging.error(f"Error creando gráfico: {e}")
            return None

class ConfluenceAnalyzer:
    """Analizador de confluencias técnicas"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.indicators_config = config.get('indicators', {})
        self.confluences_config = config.get('confluences', {})
        self.enabled = self.confluences_config.get('enabled', {})
        self.weights = self.confluences_config.get('weights', {})
        self.indicators = TechnicalIndicators()
    
    def analyze_all_confluences(self, df: pd.DataFrame, pair: str) -> Tuple[List[str], float, SignalType]:
        """Analizar todas las confluencias y calcular confianza"""
        confluences = []
        total_weight = 0
        signal_votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        # Preparar indicadores
        df = self._prepare_indicators(df)
        
        # 1. Estructura de mercado
        if self.enabled.get('market_structure', True):
            structure = self._analyze_market_structure(df)
            if structure != 'RANGING':
                confluences.append(f"Estructura {structure}")
                total_weight += self.weights.get('market_structure', 20)
                signal_votes['BUY' if structure == 'BULLISH' else 'SELL'] += 1
        
        # 2. Order Blocks
        if self.enabled.get('order_blocks', True):
            ob_signal = self._analyze_order_blocks(df)
            if ob_signal != 'HOLD':
                confluences.append(f"Order Block {ob_signal}")
                total_weight += self.weights.get('order_blocks', 15)
                signal_votes[ob_signal] += 1
        
        # 3. Fair Value Gaps
        if self.enabled.get('fair_value_gaps', True):
            fvg_signal = self._analyze_fair_value_gaps(df)
            if fvg_signal != 'HOLD':
                confluences.append(f"FVG {fvg_signal}")
                total_weight += self.weights.get('fair_value_gaps', 15)
                signal_votes[fvg_signal] += 1
        
        # 4. Liquidity Sweeps
        if self.enabled.get('liquidity_sweeps', True):
            sweep_signal = self._analyze_liquidity_sweeps(df)
            if sweep_signal != 'HOLD':
                confluences.append(f"Liquidity Sweep {sweep_signal}")
                total_weight += self.weights.get('liquidity_sweeps', 10)
                signal_votes[sweep_signal] += 1
        
        # 5. EMA Crossover
        if self.enabled.get('ema_crossover', True):
            ema_signal = self._analyze_ema_crossover(df)
            if ema_signal != 'HOLD':
                confluences.append(f"EMA Cross {ema_signal}")
                total_weight += self.weights.get('ema_crossover', 10)
                signal_votes[ema_signal] += 1
        
        # 6. RSI Divergence
        if self.enabled.get('rsi_divergence', True):
            div_signal = self._analyze_rsi_divergence(df)
            if div_signal != 'HOLD':
                confluences.append(f"RSI Divergencia {div_signal}")
                total_weight += self.weights.get('rsi_divergence', 10)
                signal_votes[div_signal] += 1
        
        # 7. Volume Analysis
        if self.enabled.get('volume_analysis', True):
            vol_signal = self._analyze_volume(df)
            if vol_signal != 'HOLD':
                confluences.append(f"Volumen {vol_signal}")
                total_weight += self.weights.get('volume_analysis', 10)
                signal_votes[vol_signal] += 1
        
        # 8. Kill Zones
        if self.enabled.get('kill_zones', True):
            if self._is_in_kill_zone():
                confluences.append("En Kill Zone ICT")
                total_weight += self.weights.get('kill_zones', 10)
        
        # Calcular confianza (0-100%)
        max_possible_weight = sum(self.weights.values())
        confidence = (total_weight / max_possible_weight) * 100 if max_possible_weight > 0 else 0
        
        # Determinar señal final basada en votos
        if signal_votes['BUY'] > signal_votes['SELL'] and signal_votes['BUY'] >= 2:
            final_signal = SignalType.BUY
        elif signal_votes['SELL'] > signal_votes['BUY'] and signal_votes['SELL'] >= 2:
            final_signal = SignalType.SELL
        else:
            final_signal = SignalType.HOLD
        
        return confluences, confidence, final_signal
    
    def _prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preparar todos los indicadores necesarios"""
        df['ema_fast'] = self.indicators.ema(df['close'], self.indicators_config.get('ema_fast', 20))
        df['ema_slow'] = self.indicators.ema(df['close'], self.indicators_config.get('ema_slow', 50))
        df['rsi'] = self.indicators.rsi(df['close'], self.indicators_config.get('rsi_period', 14))
        df['atr'] = self.indicators.atr(df['high'], df['low'], df['close'], 
                                        self.indicators_config.get('atr_period', 14))
        df['volume_ma'] = df['volume'].rolling(20).mean()
        return df
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> str:
        """Analizar estructura del mercado"""
        if len(df) < 50:
            return 'RANGING'
        
        highs = df['high'].rolling(20).max()
        lows = df['low'].rolling(20).min()
        
        recent_high = df['high'].tail(10).max()
        recent_low = df['low'].tail(10).min()
        prev_high = highs.iloc[-21:-11].max() if len(highs) > 21 else recent_high
        prev_low = lows.iloc[-21:-11].min() if len(lows) > 21 else recent_low
        
        if recent_high > prev_high and recent_low > prev_low:
            return 'BULLISH'
        elif recent_high < prev_high and recent_low < prev_low:
            return 'BEARISH'
        else:
            return 'RANGING'
    
    def _analyze_order_blocks(self, df: pd.DataFrame) -> str:
        """Analizar Order Blocks"""
        if len(df) < 30:
            return 'HOLD'
        
        current_price = df['close'].iloc[-1]
        
        for i in range(len(df)-20, len(df)-5):
            candle = df.iloc[i]
            volume_avg = df['volume_ma'].iloc[i] if 'volume_ma' in df.columns else df['volume'].mean()
            
            # Order Block bajista
            if (candle['close'] < candle['open'] and
                candle['volume'] > volume_avg * self.indicators_config.get('volume_multiplier', 1.5)):
                if candle['low'] <= current_price <= candle['high']:
                    return 'SELL'
            
            # Order Block alcista
            elif (candle['close'] > candle['open'] and
                  candle['volume'] > volume_avg * self.indicators_config.get('volume_multiplier', 1.5)):
                if candle['low'] <= current_price <= candle['high']:
                    return 'BUY'
        
        return 'HOLD'
    
    def _analyze_fair_value_gaps(self, df: pd.DataFrame) -> str:
        """Analizar Fair Value Gaps"""
        if len(df) < 10:
            return 'HOLD'
        
        current_price = df['close'].iloc[-1]
        
        for i in range(len(df)-10, len(df)-2):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            next = df.iloc[i+1]
            
            # FVG Alcista
            if prev['high'] < next['low']:
                gap_middle = (prev['high'] + next['low']) / 2
                if abs(current_price - gap_middle) < df['atr'].iloc[-1] * 0.5:
                    return 'BUY'
            
            # FVG Bajista
            elif prev['low'] > next['high']:
                gap_middle = (prev['low'] + next['high']) / 2
                if abs(current_price - gap_middle) < df['atr'].iloc[-1] * 0.5:
                    return 'SELL'
        
        return 'HOLD'
    
    def _analyze_liquidity_sweeps(self, df: pd.DataFrame) -> str:
        """Analizar barridos de liquidez"""
        if len(df) < 30:
            return 'HOLD'
        
        lookback = self.indicators_config.get('swing_lookback', 10)
        recent_high = df['high'].tail(lookback).max()
        recent_low = df['low'].tail(lookback).min()
        current_price = df['close'].iloc[-1]
        prev_high = df['high'].iloc[-lookback*2:-lookback].max()
        prev_low = df['low'].iloc[-lookback*2:-lookback].min()
        
        # Sweep de máximos con reversión
        if current_price > prev_high and df['rsi'].iloc[-1] > self.indicators_config.get('rsi_overbought', 70):
            if df['close'].iloc[-1] < df['open'].iloc[-1]:  # Vela bajista
                return 'SELL'
        
        # Sweep de mínimos con reversión
        elif current_price < prev_low and df['rsi'].iloc[-1] < self.indicators_config.get('rsi_oversold', 30):
            if df['close'].iloc[-1] > df['open'].iloc[-1]:  # Vela alcista
                return 'BUY'
        
        return 'HOLD'
    
    def _analyze_ema_crossover(self, df: pd.DataFrame) -> str:
        """Analizar cruces de EMAs"""
        if 'ema_fast' not in df.columns or 'ema_slow' not in df.columns:
            return 'HOLD'
        
        ema_fast_curr = df['ema_fast'].iloc[-1]
        ema_slow_curr = df['ema_slow'].iloc[-1]
        ema_fast_prev = df['ema_fast'].iloc[-2]
        ema_slow_prev = df['ema_slow'].iloc[-2]
        
        # Cruce alcista
        if ema_fast_prev <= ema_slow_prev and ema_fast_curr > ema_slow_curr:
            return 'BUY'
        # Cruce bajista
        elif ema_fast_prev >= ema_slow_prev and ema_fast_curr < ema_slow_curr:
            return 'SELL'
        
        return 'HOLD'
    
    def _analyze_rsi_divergence(self, df: pd.DataFrame) -> str:
        """Analizar divergencias del RSI"""
        if 'rsi' not in df.columns or len(df) < 30:
            return 'HOLD'
        
        divergence = self.indicators.detect_divergence(
            df['close'], 
            df['rsi'], 
            self.indicators_config.get('swing_lookback', 10)
        )
        
        if divergence == 'bullish':
            return 'BUY'
        elif divergence == 'bearish':
            return 'SELL'
        
        return 'HOLD'
    
    def _analyze_volume(self, df: pd.DataFrame) -> str:
        """Analizar patrones de volumen"""
        if 'volume_ma' not in df.columns or len(df) < 5:
            return 'HOLD'
        
        current_vol = df['volume'].iloc[-1]
        vol_ma = df['volume_ma'].iloc[-1]
        price_change = df['close'].iloc[-1] - df['close'].iloc[-2]
        
        multiplier = self.indicators_config.get('volume_multiplier', 1.5)
        
        # Alto volumen con movimiento alcista
        if current_vol > vol_ma * multiplier and price_change > 0:
            return 'BUY'
        # Alto volumen con movimiento bajista
        elif current_vol > vol_ma * multiplier and price_change < 0:
            return 'SELL'
        
        return 'HOLD'
    
    def _is_in_kill_zone(self) -> bool:
        """Verificar si estamos en una Kill Zone ICT"""
        current_time = datetime.now().time()
        
        kill_zones = [
            (time(1, 0), time(4, 0)),    # Asia Kill Zone
            (time(6, 0), time(9, 0)),    # London Kill Zone
            (time(13, 30), time(16, 30)) # NY Kill Zone
        ]
        
        for start, end in kill_zones:
            if start <= current_time <= end:
                return True
        return False

class EmailNotifier:
    """Sistema de notificaciones por email mejorado con gráficos"""
    
    def __init__(self, config: Dict):
        self.email_config = config.get('email', {})
        self.enabled = self.email_config.get('enabled', False)
        
    def send_signal_alert(self, signal: TradingSignal) -> bool:
        """Enviar alerta de señal con gráfico adjunto"""
        if not self.enabled:
            logging.warning("Email no habilitado en configuración")
            return False
        
        try:
            # Crear mensaje
            msg = MIMEMultipart('related')
            msg['From'] = self.email_config['email']
            msg['To'] = self.email_config['recipient']
            msg['Subject'] = f"🎯 Señal Trading - {signal.pair} {signal.signal.value} - {signal.confidence:.0f}% Confianza"
            
            # Crear HTML del email
            html_body = self._create_email_html(signal)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Adjuntar gráfico si existe
            if signal.chart_image:
                img = MIMEImage(signal.chart_image)
                img.add_header('Content-ID', '<chart>')
                msg.attach(img)
            
            # Enviar email
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['port']) as server:
                server.starttls()
                server.login(self.email_config['email'], self.email_config['password'])
                server.send_message(msg)
            
            logging.info(f"✅ Email enviado para señal {signal.pair}")
            return True
            
        except Exception as e:
            logging.error(f"Error enviando email: {e}")
            return False
    
    def _create_email_html(self, signal: TradingSignal) -> str:
        """Crear HTML formateado para el email"""
        direction_color = "#28a745" if signal.signal == SignalType.BUY else "#dc3545"
        direction_emoji = "🟢" if signal.signal == SignalType.BUY else "🔴"
        
        # Calcular ganancia potencial
        potential_profit = signal.risk_amount * signal.risk_reward
        
        # Formatear confluencias
        confluences_html = "<ul>" + "".join([f"<li>{c}</li>" for c in signal.confluences]) + "</ul>"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 20px; border-radius: 10px; }}
                .signal-box {{ background: #f8f9fa; padding: 20px; border-radius: 8px; 
                              margin: 20px 0; border-left: 4px solid {direction_color}; }}
                .metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
                .metric {{ background: white; padding: 15px; border-radius: 5px; }}
                .metric-label {{ color: #6c757d; font-size: 12px; }}
                .metric-value {{ font-size: 18px; font-weight: bold; color: #333; }}
                .confluences {{ background: #e8f5e9; padding: 15px; border-radius: 5px; 
                               margin: 15px 0; }}
                .warning {{ background: #fff3cd; padding: 15px; border-radius: 5px; 
                           border-left: 4px solid #ffc107; margin-top: 20px; }}
                .chart-container {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>{direction_emoji} Señal de Trading - {signal.pair}</h2>
                <p>Generada: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="signal-box">
                <h3 style="color: {direction_color};">Dirección: {signal.signal.value}</h3>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Entrada</div>
                        <div class="metric-value">{signal.entry_price:.5f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Stop Loss</div>
                        <div class="metric-value" style="color: #dc3545;">{signal.stop_loss:.5f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Take Profit</div>
                        <div class="metric-value" style="color: #28a745;">{signal.take_profit:.5f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Risk/Reward</div>
                        <div class="metric-value">{signal.risk_reward:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Tamaño Posición</div>
                        <div class="metric-value">{signal.position_size:.3f} lotes</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Riesgo ($)</div>
                        <div class="metric-value" style="color: #dc3545;">${signal.risk_amount:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Ganancia Potencial ($)</div>
                        <div class="metric-value" style="color: #28a745;">${potential_profit:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Confianza</div>
                        <div class="metric-value">{signal.confidence:.1f}%</div>
                    </div>
                </div>
            </div>
            
            <div class="confluences">
                <h4>📊 Confluencias Detectadas ({len(signal.confluences)})</h4>
                {confluences_html}
            </div>
            
            {"<div class='chart-container'><h4>📈 Gráfico de Análisis</h4><img src='cid:chart' width='800'></div>" if signal.chart_image else ""}
            
            <div class="warning">
                <h4>⚠️ Recordatorio Importante</h4>
                <ul>
                    <li>Verifique las condiciones del mercado antes de ejecutar</li>
                    <li>El riesgo está calculado al 1% del balance</li>
                    <li>Confirme el spread actual antes de entrar</li>
                    <li>Esta señal es solo una sugerencia, opere bajo su propio riesgo</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html

class ICTTradingSystem:
    """Sistema principal de trading ICT mejorado"""
    
    def __init__(self):
        # Cargar configuración
        self.config = ConfigManager.load_config()
        
        # Inicializar componentes
        self.mt5_connector = MT5AutoConnector(self.config)
        self.confluence_analyzer = ConfluenceAnalyzer(self.config)
        self.email_notifier = EmailNotifier(self.config)
        self.chart_generator = ChartGenerator()
        self.indicators = TechnicalIndicators()
        
        # Estado del sistema
        self.balance = self.config['trading']['default_balance']
        self.last_signal_time = {}
        
    def initialize(self) -> bool:
        """Inicializar el sistema completo"""
        logging.info("=" * 60)
        logging.info("🚀 Inicializando Sistema de Trading ICT")
        logging.info("=" * 60)
        
        # Conectar a MT5
        if self.config['mt5']['auto_connect']:
            logging.info("Conectando a MetaTrader 5...")
            if self.mt5_connector.connect_with_retry():
                self.balance = self.mt5_connector.get_balance()
                if self.balance > 0:
                    logging.info(f"✅ Balance actualizado desde MT5: ${self.balance:,.2f}")
                else:
                    logging.warning(f"⚠️ No se pudo obtener balance de MT5, usando default: ${self.balance:,.2f}")
            else:
                logging.warning("⚠️ No se pudo conectar a MT5, continuando en modo simulación")
        
        # Verificar configuración de email
        if self.email_notifier.enabled:
            logging.info(f"📧 Email configurado para: {self.config['email']['recipient']}")
        else:
            logging.warning("📧 Email no configurado")
        
        logging.info("✅ Sistema inicializado correctamente")
        return True
    
    def get_market_data(self, symbol: str, timeframe: int = 5) -> pd.DataFrame:
        """Obtener datos del mercado desde MT5 o simulados"""
        if self.mt5_connector.connected and MT5_AVAILABLE:
            try:
                # Mapear timeframe a constante MT5
                tf_map = {
                    1: mt5.TIMEFRAME_M1,
                    5: mt5.TIMEFRAME_M5,
                    15: mt5.TIMEFRAME_M15,
                    30: mt5.TIMEFRAME_M30,
                    60: mt5.TIMEFRAME_H1
                }
                mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
                
                # Obtener datos
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, 1000)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('timestamp', inplace=True)
                    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
                    logging.info(f"✅ Datos reales obtenidos para {symbol}")
                    return df
            except Exception as e:
                logging.error(f"Error obteniendo datos de MT5: {e}")
        
        # Generar datos simulados como fallback
        logging.info(f"📊 Usando datos simulados para {symbol}")
        return self._generate_simulated_data(symbol)
    
    def _generate_simulated_data(self, symbol: str) -> pd.DataFrame:
        """Generar datos simulados para pruebas"""
        np.random.seed(42)
        
        # Precios base por símbolo
        base_prices = {
            'EURUSD': 1.0950,
            'GBPUSD': 1.2650,
            'USDJPY': 150.00,
            'XAUUSD': 2050.00
        }
        base_price = base_prices.get(symbol, 1.0000)
        
        # Generar 1000 velas de 5 minutos
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='5min')
        
        # Simular movimiento de precio realista
        returns = np.random.normal(0, 0.001, 1000)
        price_series = base_price * (1 + returns).cumprod()
        
        data = []
        for i, date in enumerate(dates):
            if i == 0:
                open_price = base_price
            else:
                open_price = data[i-1]['close']
            
            close_price = price_series[i]
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0005)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0005)))
            volume = np.random.exponential(1000)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def calculate_position_size(self, entry: float, stop_loss: float, symbol: str) -> float:
        """Calcular tamaño de posición basado en riesgo del 1%"""
        risk_percentage = self.config['trading']['risk_percentage'] / 100
        risk_amount = self.balance * risk_percentage
        
        # Obtener configuración del par
        pair_config = self.config['pairs'].get(symbol, {})
        min_lot = pair_config.get('lot_size', 0.01)
        
        # Calcular distancia en pips/puntos
        if 'JPY' in symbol:
            pip_value = 0.01
            pips = abs(entry - stop_loss) / pip_value
            # Para JPY, 1 pip = $10 por lote estándar
            position_size = risk_amount / (pips * 10)
        elif 'XAU' in symbol:
            # Para oro, movimiento en dólares
            dollar_move = abs(entry - stop_loss)
            # 1 lote = 100 onzas, $1 movimiento = $100 por lote
            position_size = risk_amount / (dollar_move * 100)
        else:
            pip_value = 0.0001
            pips = abs(entry - stop_loss) / pip_value
            # Para pares principales, 1 pip = $10 por lote estándar
            position_size = risk_amount / (pips * 10)
        
        # Redondear al tamaño de lote mínimo
        position_size = max(min_lot, round(position_size / min_lot) * min_lot)
        
        # Limitar a máximo 10% del balance
        max_position = self.balance * 0.1 / 10000  # Convertir a lotes
        position_size = min(position_size, max_position)
        
        return position_size
    
    def generate_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generar señal de trading si hay suficientes confluencias"""
        try:
            # Obtener datos
            df = self.get_market_data(symbol)
            if df is None or len(df) < 100:
                logging.warning(f"Datos insuficientes para {symbol}")
                return None
            
            # Analizar confluencias
            confluences, confidence, signal_type = self.confluence_analyzer.analyze_all_confluences(df, symbol)
            
            # Verificar confianza mínima
            min_confidence = self.config['trading']['min_confidence']
            if confidence < min_confidence or signal_type == SignalType.HOLD:
                logging.info(f"Confianza insuficiente para {symbol}: {confidence:.1f}% < {min_confidence}%")
                return None
            
            # Evitar señales duplicadas (1 por hora por símbolo)
            current_time = datetime.now()
            if symbol in self.last_signal_time:
                time_diff = (current_time - self.last_signal_time[symbol]).total_seconds()
                if time_diff < 3600:  # 1 hora
                    logging.info(f"Señal reciente para {symbol}, esperando...")
                    return None
            
            # Calcular niveles de entrada, SL y TP
            current_price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else (df['high'] - df['low']).mean()
            
            pair_config = self.config['pairs'].get(symbol, {})
            spread = pair_config.get('spread', 0.0002)
            min_rr = pair_config.get('min_rr', 2.0)
            
            if signal_type == SignalType.BUY:
                entry = current_price + spread  # Ajustar por spread
                stop_loss = entry - (atr * 1.5)
                take_profit = entry + (atr * 1.5 * min_rr)
            else:  # SELL
                entry = current_price - spread
                stop_loss = entry + (atr * 1.5)
                take_profit = entry - (atr * 1.5 * min_rr)
            
            # Calcular Risk/Reward
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            risk_reward = reward / risk if risk > 0 else 0
            
            # Verificar RR mínimo
            if risk_reward < self.config['trading']['min_risk_reward']:
                logging.info(f"R/R insuficiente para {symbol}: {risk_reward:.2f}")
                return None
            
            # Calcular tamaño de posición
            position_size = self.calculate_position_size(entry, stop_loss, symbol)
            risk_amount = self.balance * (self.config['trading']['risk_percentage'] / 100)
            
            # Generar gráfico
            chart_image = self.chart_generator.create_signal_chart(
                df, 
                TradingSignal(
                    timestamp=current_time,
                    pair=symbol,
                    signal=signal_type,
                    entry_price=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward=risk_reward,
                    confidence=confidence,
                    confluences=confluences,
                    position_size=position_size,
                    risk_amount=risk_amount
                ),
                self.config['indicators']
            )
            
            # Crear señal
            signal = TradingSignal(
                timestamp=current_time,
                pair=symbol,
                signal=signal_type,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=risk_reward,
                confidence=confidence,
                confluences=confluences,
                position_size=position_size,
                risk_amount=risk_amount,
                chart_image=chart_image
            )
            
            # Actualizar tiempo de última señal
            self.last_signal_time[symbol] = current_time
            
            return signal
            
        except Exception as e:
            logging.error(f"Error generando señal para {symbol}: {e}")
            return None
    
    def process_symbols(self, symbols: List[str]):
        """Procesar lista de símbolos y generar señales"""
        signals_found = []
        
        for symbol in symbols:
            logging.info(f"\n📊 Analizando {symbol}...")
            
            signal = self.generate_signal(symbol)
            if signal:
                signals_found.append(signal)
                
                # Mostrar en consola
                self.display_signal_console(signal)
                
                # Enviar por email
                if self.email_notifier.enabled:
                    if self.email_notifier.send_signal_alert(signal):
                        logging.info(f"✅ Email enviado para {signal.pair}")
                    else:
                        logging.error(f"❌ Error enviando email para {signal.pair}")
            else:
                logging.info(f"❌ No hay señal para {symbol}")
        
        return signals_found
    
    def display_signal_console(self, signal: TradingSignal):
        """Mostrar señal en consola de forma clara"""
        print("\n" + "=" * 60)
        print(f"🎯 NUEVA SEÑAL DETECTADA - {signal.pair}")
        print("=" * 60)
        print(f"📅 Timestamp: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Dirección: {signal.signal.value}")
        print(f"💰 Entrada: {signal.entry_price:.5f}")
        print(f"🛑 Stop Loss: {signal.stop_loss:.5f}")
        print(f"🎯 Take Profit: {signal.take_profit:.5f}")
        print(f"📈 Risk/Reward: {signal.risk_reward:.2f}")
        print(f"📊 Tamaño Posición: {signal.position_size:.3f} lotes")
        print(f"💵 Riesgo ($): ${signal.risk_amount:.2f}")
        print(f"🔍 Confianza: {signal.confidence:.1f}%")
        print(f"\n📋 Confluencias ({len(signal.confluences)}):")
        for conf in signal.confluences:
            print(f"   ✓ {conf}")
        print("=" * 60)
    
    def run_continuous(self, symbols: List[str], interval_minutes: int = 5):
        """Ejecutar monitoreo continuo"""
        logging.info(f"\n🚀 Iniciando monitoreo continuo")
        logging.info(f"📊 Símbolos: {', '.join(symbols)}")
        logging.info(f"⏱️ Intervalo: {interval_minutes} minutos")
        logging.info(f"💰 Balance: ${self.balance:,.2f}")
        logging.info("Presione Ctrl+C para detener...\n")
        
        # Ejecutar análisis inicial
        self.process_symbols(symbols)
        
        # Programar ejecución periódica
        schedule.every(interval_minutes).minutes.do(self.process_symbols, symbols)
        
        try:
            while True:
                schedule.run_pending()
                time_module.sleep(30)
        except KeyboardInterrupt:
            logging.info("\n🛑 Monitoreo detenido por el usuario")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpiar recursos al finalizar"""
        if self.mt5_connector:
            self.mt5_connector.disconnect()
        logging.info("Sistema cerrado correctamente")

def main():
    """Función principal"""
    # Crear sistema
    system = ICTTradingSystem()
    
    # Inicializar
    if not system.initialize():
        logging.error("Error inicializando el sistema")
        return
    
    # Obtener símbolos desde configuración
    symbols = system.config['trading']['default_symbols']
    interval = system.config['trading']['default_interval']
    
    # Menú de opciones
    print("\n📊 Sistema de Trading ICT")
    print("1. Análisis único")
    print("2. Monitoreo continuo")
    print("3. Configurar sistema")
    
    choice = input("\nSeleccione opción (1-3): ")
    
    if choice == "1":
        # Análisis único
        signals = system.process_symbols(symbols)
        if signals:
            print(f"\n✅ Se encontraron {len(signals)} señales")
        else:
            print("\n❌ No se encontraron señales en este momento")
    
    elif choice == "2":
        # Monitoreo continuo
        system.run_continuous(symbols, interval)
    
    elif choice == "3":
        # Reconfigurar
        ConfigManager.create_default_config()
        print("✅ Archivo config.json regenerado. Edítelo y reinicie el programa.")
    
    else:
        print("Opción inválida")

if __name__ == "__main__":
    main()