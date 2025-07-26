import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import sys

# Configurar codificación UTF-8 para Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class BacktestTrade:
    """Clase para representar una operación en backtesting"""
    pair: str
    direction: str
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    exit_reason: str  # 'TP', 'SL', 'TIMEOUT'
    profit_loss: float
    position_size: float
    duration_hours: float

@dataclass
class PairResults:
    """Resultados de backtesting por par"""
    pair: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    max_profit: float
    max_loss: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    max_consecutive_losses: int

class ForexBacktester:
    def __init__(self, config_file: str = 'config.json'):
        """Inicializar el backtester"""
        self.config = self.load_config(config_file)
        self.mt5_connected = False
        self.trades = []
        self.equity_curve = []
        self.initial_balance = 10000  # Balance inicial por defecto
        
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
                "rsi_oversold": 20,
                "rsi_overbought": 80,
                "zscore_window": 40,
                "zscore_threshold": 1.8,
                "atr_period": 21,
                "atr_multiplier": 1.8,
                "min_confidence": 0.65,
                "trend_filter": True,
                "trend_period": 200
            },
            "risk": {
                "max_risk_per_trade": 0.02,
                "min_risk_reward": 1.5,
                "max_spread_pips": 3.0,
                "use_equity_for_risk": True,
                "max_position_size": 10.0,
                "min_account_balance": 1000
            },
            "backtest": {
                "initial_balance": 10000,
                "max_trade_duration_hours": 168,  # 1 semana máximo
                "spread_cost_pips": 1.5,  # Costo promedio de spread
                "commission_per_lot": 7.0  # Comisión por lote
            }
        }
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
        except FileNotFoundError:
            logging.warning(f"Archivo {config_file} no encontrado, usando configuración por defecto")
            return default_config
    
    def connect_mt5(self) -> bool:
        """Conectar a MetaTrader 5"""
        if not mt5.initialize():
            logging.error(f"Error inicializando MT5: {mt5.last_error()}")
            return False
        
        logging.info("Conectado a MT5 para obtener datos históricos")
        self.mt5_connected = True
        return True
    
    def get_historical_data(self, symbol: str, timeframe=mt5.TIMEFRAME_M5, # TEMPORALIDAD 15M
                           start_date: datetime = None, end_date: datetime = None) -> Optional[pd.DataFrame]:
        """Obtener datos históricos desde MT5"""
        if not self.mt5_connected:
            if not self.connect_mt5():
                return None
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)  # 2 años por defecto
        if end_date is None:
            end_date = datetime.now()
        
        # Obtener datos históricos
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None:
            logging.error(f"No se pudieron obtener datos para {symbol}: {mt5.last_error()}")
            return None
        
        # Convertir a DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Mapear columnas correctamente
        if len(df.columns) >= 5:
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] + [f'Extra_{i}' for i in range(len(df.columns) - 5)]
        
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
    
    def get_position_size(self, current_balance: float, risk_amount: float, 
                         entry_price: float, stop_loss: float, symbol: str) -> float:
        """Calcular tamaño de posición"""
        risk_distance = abs(entry_price - stop_loss)
        
        # Determinar el valor del pip
        if 'JPY' in symbol:
            pip_size = 0.01
            pip_value_per_lot = pip_size * 100000 / entry_price if symbol.startswith('USD') else pip_size * 100000
        else:
            pip_size = 0.0001
            pip_value_per_lot = pip_size * 100000 if symbol.endswith('USD') else pip_size * 100000 / entry_price
        
        # Calcular pips en riesgo
        pips_at_risk = risk_distance / pip_size
        
        if pips_at_risk > 0 and pip_value_per_lot > 0:
            position_size = risk_amount / (pips_at_risk * pip_value_per_lot)
            # Limitar tamaño máximo
            position_size = min(position_size, self.config['risk']['max_position_size'])
            # Redondear a 0.01 lotes
            position_size = round(position_size, 2)
            return max(0.01, position_size)
        
        return 0.01
    
    def calculate_profit_loss(self, trade: dict, symbol: str) -> float:
        """Calcular profit/loss de una operación"""
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        position_size = trade['position_size']
        direction = trade['direction']
        
        # Calcular diferencia de precio
        if direction == 'BUY':
            price_diff = exit_price - entry_price
        else:  # SELL
            price_diff = entry_price - exit_price
        
        # Calcular profit/loss en dinero
        if 'JPY' in symbol:
            profit_loss = price_diff * position_size * 100000
        else:
            profit_loss = price_diff * position_size * 100000
        
        # Restar costos de spread y comisión
        spread_cost = self.config['backtest']['spread_cost_pips'] * position_size * 10
        commission = self.config['backtest']['commission_per_lot'] * position_size
        
        return profit_loss - spread_cost - commission
    
    def find_exit_point(self, df: pd.DataFrame, entry_idx: int, trade_params: dict) -> Tuple[int, float, str]:
        """Encontrar el punto de salida de la operación"""
        direction = trade_params['direction']
        stop_loss = trade_params['stop_loss']
        take_profit = trade_params['take_profit']
        max_duration = self.config['backtest']['max_trade_duration_hours']
        
        start_time = df.index[entry_idx]
        max_time = start_time + timedelta(hours=max_duration)
        
        for i in range(entry_idx + 1, len(df)):
            current = df.iloc[i]
            current_time = df.index[i]
            
            # Verificar timeout
            if current_time >= max_time:
                return i, current['Close'], 'TIMEOUT'
            
            # Verificar stop loss y take profit
            if direction == 'BUY':
                if current['Low'] <= stop_loss:
                    return i, stop_loss, 'SL'
                elif current['High'] >= take_profit:
                    return i, take_profit, 'TP'
            else:  # SELL
                if current['High'] >= stop_loss:
                    return i, stop_loss, 'SL'
                elif current['Low'] <= take_profit:
                    return i, take_profit, 'TP'
        
        # Si llegamos al final sin salida, cerrar al último precio
        return len(df) - 1, df.iloc[-1]['Close'], 'END'
    
    def generate_signals(self, df: pd.DataFrame, symbol: str) -> List[dict]:
        """Generar señales de trading basadas en la estrategia"""
        signals = []
        strategy = self.config['strategy']
        
        # Necesitamos suficientes datos para los indicadores
        start_idx = max(strategy['trend_period'], strategy['zscore_window'], strategy['rsi_period']) + 1
        
        for i in range(start_idx, len(df) - 1):  # -1 para poder encontrar salida
            current = df.iloc[i]
            
            # Verificar que todos los indicadores estén disponibles
            required_indicators = ['ZScore', 'RSI', 'ATR', 'SMA']
            if any(pd.isna(current[indicator]) for indicator in required_indicators):
                continue
            
            signal = None
            
            # Señal de COMPRA
            if (current['ZScore'] < -strategy['zscore_threshold'] and 
                current['RSI'] < strategy['rsi_oversold']):
                
                confidence = min(0.9, abs(current['ZScore']) / 3.0 + (80 - current['RSI']) / 100)
                
                if strategy['trend_filter'] and 'Trend' in df.columns and current['Trend'] < 0:
                    confidence *= 0.7
                
                if confidence >= strategy['min_confidence']:
                    entry_price = current['Close']  # Simplificado para backtesting
                    atr = current['ATR']
                    stop_loss = entry_price - (atr * strategy['atr_multiplier'])
                    take_profit = current['SMA']
                    
                    risk_distance = entry_price - stop_loss
                    reward_distance = take_profit - entry_price
                    risk_reward = reward_distance / risk_distance if risk_distance > 0 else 0
                    
                    if risk_reward >= self.config['risk']['min_risk_reward']:
                        signal = {
                            'index': i,
                            'direction': 'BUY',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'confidence': confidence,
                            'atr': atr
                        }
            
            # Señal de VENTA
            elif (current['ZScore'] > strategy['zscore_threshold'] and 
                  current['RSI'] > strategy['rsi_overbought']):
                
                confidence = min(0.9, abs(current['ZScore']) / 3.0 + (current['RSI'] - 20) / 100)
                
                if strategy['trend_filter'] and 'Trend' in df.columns and current['Trend'] > 0:
                    confidence *= 0.7
                
                if confidence >= strategy['min_confidence']:
                    entry_price = current['Close']  # Simplificado para backtesting
                    atr = current['ATR']
                    stop_loss = entry_price + (atr * strategy['atr_multiplier'])
                    take_profit = current['SMA']
                    
                    risk_distance = stop_loss - entry_price
                    reward_distance = entry_price - take_profit
                    risk_reward = reward_distance / risk_distance if risk_distance > 0 else 0
                    
                    if risk_reward >= self.config['risk']['min_risk_reward']:
                        signal = {
                            'index': i,
                            'direction': 'SELL',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'confidence': confidence,
                            'atr': atr
                        }
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def backtest_pair(self, symbol: str, start_date: datetime, end_date: datetime) -> PairResults:
        """Realizar backtesting para un par específico"""
        logging.info(f"Backtesting {symbol}...")
        
        # Obtener datos históricos
        df = self.get_historical_data(symbol, mt5.TIMEFRAME_M15, start_date, end_date) # TEMPORALIDAD H1
        if df is None:
            logging.error(f"No se pudieron obtener datos para {symbol}")
            return PairResults(symbol, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        
        # Calcular indicadores
        df = self.calculate_indicators(df)
        
        # Generar señales
        signals = self.generate_signals(df, symbol)
        
        if not signals:
            logging.warning(f"No se generaron señales para {symbol}")
            return PairResults(symbol, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        
        # Procesar cada señal
        trades = []
        current_balance = self.initial_balance
        
        for signal in signals:
            # Calcular tamaño de posición
            risk_amount = current_balance * self.config['risk']['max_risk_per_trade']
            position_size = self.get_position_size(
                current_balance, risk_amount, 
                signal['entry_price'], signal['stop_loss'], symbol
            )
            
            # Encontrar punto de salida
            exit_idx, exit_price, exit_reason = self.find_exit_point(df, signal['index'], signal)
            
            # Crear registro de operación
            trade = {
                'entry_price': signal['entry_price'],
                'exit_price': exit_price,
                'position_size': position_size,
                'direction': signal['direction']
            }
            
            # Calcular profit/loss
            profit_loss = self.calculate_profit_loss(trade, symbol)
            
            # Crear objeto BacktestTrade
            backtest_trade = BacktestTrade(
                pair=symbol,
                direction=signal['direction'],
                entry_price=signal['entry_price'],
                entry_time=df.index[signal['index']],
                exit_price=exit_price,
                exit_time=df.index[exit_idx],
                exit_reason=exit_reason,
                profit_loss=profit_loss,
                position_size=position_size,
                duration_hours=(df.index[exit_idx] - df.index[signal['index']]).total_seconds() / 3600
            )
            
            trades.append(backtest_trade)
            current_balance += profit_loss
        
        # Calcular estadísticas
        return self.calculate_pair_statistics(trades, symbol)
    
    def calculate_pair_statistics(self, trades: List[BacktestTrade], symbol: str) -> PairResults:
        """Calcular estadísticas para un par"""
        if not trades:
            return PairResults(symbol, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.profit_loss > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        profits = [t.profit_loss for t in trades if t.profit_loss > 0]
        losses = [t.profit_loss for t in trades if t.profit_loss < 0]
        
        total_profit = sum(t.profit_loss for t in trades)
        max_profit = max(profits) if profits else 0
        max_loss = min(losses) if losses else 0
        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(profits) if profits else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Máximo de pérdidas consecutivas
        consecutive_losses = 0
        max_consecutive_losses = 0
        for trade in trades:
            if trade.profit_loss < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return PairResults(
            pair=symbol,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_profit=total_profit,
            max_profit=max_profit,
            max_loss=max_loss,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_consecutive_losses=max_consecutive_losses
        )
    
    def run_backtest(self, years: int = 2) -> Dict[str, PairResults]:
        """Ejecutar backtesting completo"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        self.initial_balance = self.config.get('backtest', {}).get('initial_balance', 10000)
        
        logging.info(f"Iniciando backtesting desde {start_date.strftime('%Y-%m-%d')} hasta {end_date.strftime('%Y-%m-%d')}")
        logging.info(f"Balance inicial: ${self.initial_balance:,.2f}")
        
        results = {}
        total_balance = self.initial_balance
        all_trades = []
        
        for symbol, settings in self.config['pairs'].items():
            if not settings.get('enabled', True):
                continue
            
            pair_result = self.backtest_pair(symbol, start_date, end_date)
            results[symbol] = pair_result
            total_balance += pair_result.total_profit
        
        # Calcular drawdown máximo
        max_drawdown = self.calculate_max_drawdown(results)
        
        # Mostrar resumen
        self.print_summary(results, total_balance, max_drawdown, years)
        
        return results
    
    def calculate_max_drawdown(self, results: Dict[str, PairResults]) -> float:
        """Calcular drawdown máximo aproximado"""
        # Simplificación: usar la suma de las pérdidas máximas de cada par
        max_loss_per_pair = [abs(result.max_loss) for result in results.values()]
        return sum(max_loss_per_pair)
    
    def print_summary(self, results: Dict[str, PairResults], final_balance: float, max_drawdown: float, years: int):
        """Imprimir resumen de resultados"""
        print("\n" + "="*80)
        print("                    RESUMEN DE BACKTESTING")
        print("="*80)
        print(f"📅 Período: {years} año{'s' if years != 1 else ''}")
        print(f"💰 Balance inicial: ${self.initial_balance:,.2f}")
        print(f"💰 Balance final: ${final_balance:,.2f}")
        print(f"📈 Ganancia total: ${final_balance - self.initial_balance:,.2f} ({((final_balance/self.initial_balance - 1) * 100):+.2f}%)")
        print(f"📊 Rendimiento anualizado: {(((final_balance/self.initial_balance) ** (1/years)) - 1) * 100:+.2f}%")
        print(f"📉 Drawdown máximo estimado: ${max_drawdown:,.2f}")
        print("\n" + "="*80)
        print("                    RESULTADOS POR PAR")
        print("="*80)
        print(f"{'Par':<8} {'Trades':<7} {'Win Rate':<9} {'Profit':<12} {'Max DD':<10}")
        print("-"*80)
        
        total_trades = 0
        total_winning = 0
        
        for symbol, result in results.items():
            if result.total_trades > 0:
                print(f"{symbol:<8} {result.total_trades:<7} {result.win_rate*100:>6.1f}%   "
                      f"${result.total_profit:>9.2f}   ${abs(result.max_loss):>7.2f}")
                total_trades += result.total_trades
                total_winning += result.winning_trades
        
        print("-"*80)
        overall_winrate = (total_winning / total_trades * 100) if total_trades > 0 else 0
        print(f"{'TOTAL':<8} {total_trades:<7} {overall_winrate:>6.1f}%   "
              f"${final_balance - self.initial_balance:>9.2f}   ${max_drawdown:>7.2f}")
        print("="*80)
        
        # Estadísticas adicionales
        profitable_pairs = len([r for r in results.values() if r.total_profit > 0])
        print(f"\n📊 Estadísticas adicionales:")
        print(f"   • Pares rentables: {profitable_pairs}/{len(results)}")
        print(f"   • Operaciones totales: {total_trades}")
        print(f"   • Win rate general: {overall_winrate:.1f}%")
        
        if total_trades > 0:
            avg_profit_per_trade = (final_balance - self.initial_balance) / total_trades
            print(f"   • Ganancia promedio por operación: ${avg_profit_per_trade:.2f}")
            print(f"   • Operaciones por año: {total_trades / years:.1f}")

def get_user_years() -> int:
    """Solicitar al usuario cuántos años desea para el backtesting"""
    while True:
        try:
            years_input = input("📅 Cantidad de años para backtesting (1-10, Enter para 2 años): ").strip()
            
            # Si no ingresa nada, usar 2 años por defecto
            if not years_input:
                return 2
            
            years = int(years_input)
            
            # Validar rango
            if 1 <= years <= 10:
                return years
            else:
                print("❌ Por favor ingrese un número entre 1 y 10 años")
                
        except ValueError:
            print("❌ Por favor ingrese un número válido")

def main():
    """Función principal"""
    print("🔬 Backtester de Estrategia Forex")
    print("="*50)
    
    # Crear backtester
    backtester = ForexBacktester()
    
    # Configurar balance inicial
    initial_balance = input("💰 Balance inicial (Enter para $10,000): ").strip()
    if initial_balance:
        try:
            backtester.initial_balance = float(initial_balance)
            backtester.config['backtest']['initial_balance'] = backtester.initial_balance
        except ValueError:
            print("❌ Valor inválido, usando $10,000")
    
    # Solicitar años para backtesting
    years = get_user_years()
    
    # Ejecutar backtesting
    print(f"\n🚀 Iniciando backtesting de {years} año{'s' if years != 1 else ''} con balance inicial de ${backtester.initial_balance:,.2f}")
    print("⏳ Esto puede tomar varios minutos...")
    
    try:
        results = backtester.run_backtest(years=years)
        
        print("\n✅ Backtesting completado!")
        print("\n💡 Notas importantes:")
        print("   • Los resultados son aproximados y no incluyen slippage")
        print("   • Se asume ejecución perfecta de órdenes")
        print("   • Los costos de spread y comisión están incluidos")
        print("   • El drawdown es una estimación basada en pérdidas máximas")
        
    except KeyboardInterrupt:
        print("\n⏹️ Backtesting interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante el backtesting: {e}")
        logging.error(f"Error en backtesting: {e}")

if __name__ == "__main__":
    main()