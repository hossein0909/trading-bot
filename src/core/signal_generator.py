# src/core/signal_generator.py
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Tuple

class QuotexSignalGenerator:
    def __init__(self):
        self.setup_logging()
        self.asset_groups = {
            'forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X'],
            'crypto': ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD'],
            'stocks': ['TSLA', 'AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'NVDA']
        }
        
        self.timeframes = {
            '1m': '1m',
            '5m': '5m', 
            '15m': '15m'
        }
        
        print("✅ Quotex Signal Generator Initialized")
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_quotex_assets_data(self):
        """Get market data for all Quotex assets"""
        all_assets_data = {}
        
        for category, assets in self.asset_groups.items():
            print(f"\n📊 Fetching {category.upper()} data...")
            
            for asset in assets:
                try:
                    # Get data for multiple timeframes
                    asset_data = {}
                    
                    for tf_name, tf_value in self.timeframes.items():
                        ticker = yf.Ticker(asset)
                        data = ticker.history(period='1d', interval=tf_value)
                        
                        if not data.empty:
                            analyzed_data = self.analyze_asset_data(data, asset, tf_name)
                            asset_data[tf_name] = analyzed_data
                    
                    if asset_data:
                        all_assets_data[asset] = {
                            'category': category,
                            'timeframes': asset_data,
                            'current_price': float(data['Close'].iloc[-1]) if not data.empty else 0
                        }
                        
                        print(f"   ✅ {asset} - ${all_assets_data[asset]['current_price']:.2f}")
                        
                except Exception as e:
                    print(f"   ❌ Error fetching {asset}: {e}")
        
        return all_assets_data
    
    def analyze_asset_data(self, data, asset, timeframe):
        """Analyze asset data for signal generation"""
        if data.empty:
            return None
        
        df = data.copy()
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Generate signals for this timeframe
        signals = self.generate_timeframe_signals(df, asset, timeframe)
        
        return {
            'data': df,
            'signals': signals,
            'trend': self.determine_trend(df),
            'volatility': self.calculate_volatility(df),
            'support_resistance': self.find_support_resistance(df)
        }
    
    def calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        # Price moving averages
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic
        df['stoch_k'] = 100 * (df['Close'] - df['Low'].rolling(14).min()) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min())
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Volume indicators
        df['volume_sma'] = df['Volume'].rolling(10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        return df.dropna()
    
    def generate_timeframe_signals(self, df, asset, timeframe):
        """Generate trading signals for specific timeframe"""
        if df.empty:
            return {}
        
        current_data = df.iloc[-1]
        signals = {}
        
        # 1. Trend Analysis
        trend_strength = self.analyze_trend_strength(df)
        signals['trend'] = trend_strength
        
        # 2. Momentum Signals
        signals['rsi_signal'] = self.get_rsi_signal(current_data['rsi'])
        signals['macd_signal'] = self.get_macd_signal(current_data['macd'], current_data['macd_signal'])
        signals['stoch_signal'] = self.get_stoch_signal(current_data['stoch_k'], current_data['stoch_d'])
        
        # 3. Volatility Signals
        signals['bb_signal'] = self.get_bb_signal(current_data['bb_position'])
        signals['volatility'] = self.calculate_volatility(df)
        
        # 4. Volume Analysis
        signals['volume_signal'] = self.get_volume_signal(current_data['volume_ratio'])
        
        # 5. Combined Signal
        signals['combined_signal'] = self.calculate_combined_signal(signals, timeframe)
        signals['confidence'] = self.calculate_signal_confidence(signals, timeframe)
        
        return signals
    
    def analyze_trend_strength(self, df):
        """Analyze trend strength and direction"""
        if len(df) < 20:
            return {'direction': 'NEUTRAL', 'strength': 0}
        
        current_price = df['Close'].iloc[-1]
        sma_5 = df['sma_5'].iloc[-1]
        sma_10 = df['sma_10'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        
        # Trend direction
        if current_price > sma_5 > sma_10 > sma_20:
            direction = 'STRONG_BULL'
            strength = 0.9
        elif current_price < sma_5 < sma_10 < sma_20:
            direction = 'STRONG_BEAR'
            strength = 0.9
        elif current_price > sma_20:
            direction = 'BULL'
            strength = 0.7
        elif current_price < sma_20:
            direction = 'BEAR'
            strength = 0.7
        else:
            direction = 'NEUTRAL'
            strength = 0.5
        
        return {'direction': direction, 'strength': strength}
    
    def get_rsi_signal(self, rsi):
        """Generate RSI-based signal"""
        if rsi < 30:
            return {'signal': 'STRONG_BUY', 'confidence': 0.8}
        elif rsi < 40:
            return {'signal': 'BUY', 'confidence': 0.6}
        elif rsi > 70:
            return {'signal': 'STRONG_SELL', 'confidence': 0.8}
        elif rsi > 60:
            return {'signal': 'SELL', 'confidence': 0.6}
        else:
            return {'signal': 'NEUTRAL', 'confidence': 0.3}
    
    def get_macd_signal(self, macd, macd_signal):
        """Generate MACD-based signal"""
        if macd > macd_signal and macd > 0:
            return {'signal': 'STRONG_BUY', 'confidence': 0.8}
        elif macd > macd_signal:
            return {'signal': 'BUY', 'confidence': 0.6}
        elif macd < macd_signal and macd < 0:
            return {'signal': 'STRONG_SELL', 'confidence': 0.8}
        elif macd < macd_signal:
            return {'signal': 'SELL', 'confidence': 0.6}
        else:
            return {'signal': 'NEUTRAL', 'confidence': 0.3}
    
    def get_bb_signal(self, bb_position):
        """Generate Bollinger Bands signal"""
        if bb_position < 0.2:
            return {'signal': 'STRONG_BUY', 'confidence': 0.7}
        elif bb_position < 0.3:
            return {'signal': 'BUY', 'confidence': 0.5}
        elif bb_position > 0.8:
            return {'signal': 'STRONG_SELL', 'confidence': 0.7}
        elif bb_position > 0.7:
            return {'signal': 'SELL', 'confidence': 0.5}
        else:
            return {'signal': 'NEUTRAL', 'confidence': 0.3}
    
    def get_stoch_signal(self, stoch_k, stoch_d):
        """Generate Stochastic signal"""
        if stoch_k < 20 and stoch_d < 20:
            return {'signal': 'STRONG_BUY', 'confidence': 0.7}
        elif stoch_k < 30 and stoch_d < 30:
            return {'signal': 'BUY', 'confidence': 0.5}
        elif stoch_k > 80 and stoch_d > 80:
            return {'signal': 'STRONG_SELL', 'confidence': 0.7}
        elif stoch_k > 70 and stoch_d > 70:
            return {'signal': 'SELL', 'confidence': 0.5}
        else:
            return {'signal': 'NEUTRAL', 'confidence': 0.3}
    
    def get_volume_signal(self, volume_ratio):
        """Generate volume-based signal"""
        if volume_ratio > 2.0:
            return {'signal': 'HIGH_VOLUME', 'confidence': 0.8}
        elif volume_ratio > 1.5:
            return {'signal': 'MEDIUM_VOLUME', 'confidence': 0.6}
        else:
            return {'signal': 'LOW_VOLUME', 'confidence': 0.3}
    
    def calculate_combined_signal(self, signals, timeframe):
        """Calculate combined trading signal"""
        buy_signals = 0
        sell_signals = 0
        total_confidence = 0
        signal_count = 0
        
        signal_components = ['rsi_signal', 'macd_signal', 'bb_signal', 'stoch_signal']
        
        for component in signal_components:
            if component in signals:
                signal_data = signals[component]
                if 'BUY' in signal_data['signal']:
                    buy_signals += 1
                elif 'SELL' in signal_data['signal']:
                    sell_signals += 1
                total_confidence += signal_data['confidence']
                signal_count += 1
        
        # Consider trend
        trend = signals.get('trend', {})
        if trend.get('direction') in ['STRONG_BULL', 'BULL']:
            buy_signals += 2
        elif trend.get('direction') in ['STRONG_BEAR', 'BEAR']:
            sell_signals += 2
        
        # Determine final signal
        if buy_signals > sell_signals + 2:
            return 'STRONG_BUY'
        elif buy_signals > sell_signals:
            return 'BUY'
        elif sell_signals > buy_signals + 2:
            return 'STRONG_SELL'
        elif sell_signals > buy_signals:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def calculate_signal_confidence(self, signals, timeframe):
        """Calculate overall signal confidence"""
        confidences = []
        
        signal_components = ['rsi_signal', 'macd_signal', 'bb_signal', 'stoch_signal']
        
        for component in signal_components:
            if component in signals:
                confidences.append(signals[component]['confidence'])
        
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            
            # Adjust confidence based on timeframe
            if timeframe == '1m':
                return avg_confidence * 0.7  # Lower confidence for 1min
            elif timeframe == '5m':
                return avg_confidence * 0.8
            else:  # 15m
                return avg_confidence * 0.9
        else:
            return 0.3
    
    def determine_trend(self, df):
        """Determine market trend"""
        if len(df) < 10:
            return 'SIDEWAYS'
        
        prices = df['Close'].tail(10)
        if prices.is_monotonic_increasing:
            return 'UPTREND'
        elif prices.is_monotonic_decreasing:
            return 'DOWNTREND'
        else:
            return 'SIDEWAYS'
    
    def calculate_volatility(self, df, window=20):
        """Calculate price volatility"""
        if len(df) < window:
            return 0.1
        
        returns = df['Close'].pct_change().dropna()
        volatility = returns.rolling(window=window).std().iloc[-1]
        return volatility if not np.isnan(volatility) else 0.1
    
    def find_support_resistance(self, df, window=20):
        """Find support and resistance levels"""
        if len(df) < window:
            return {'support': 0, 'resistance': 0}
        
        recent_data = df.tail(window)
        support = recent_data['Low'].min()
        resistance = recent_data['High'].max()
        
        return {'support': support, 'resistance': resistance}
    
    def generate_quotex_signals(self):
        """Generate comprehensive signals for Quotex assets"""
        print("\n" + "="*60)
        print("🎯 GENERATING QUOTEX TRADING SIGNALS")
        print("="*60)
        
        # Get all assets data
        assets_data = self.get_quotex_assets_data()
        
        signals_report = {}
        
        for asset, data in assets_data.items():
            print(f"\n📈 Analyzing {asset}...")
            
            asset_signals = {}
            
            for timeframe, tf_data in data['timeframes'].items():
                signals = tf_data['signals']
                
                if signals['confidence'] > 0.6:  # Only show high-confidence signals
                    asset_signals[timeframe] = {
                        'signal': signals['combined_signal'],
                        'confidence': signals['confidence'],
                        'current_price': data['current_price'],
                        'trend': tf_data['trend'],
                        'timestamp': datetime.now()
                    }
                    
                    print(f"   ⏰ {timeframe}: {signals['combined_signal']} "
                          f"(Confidence: {signals['confidence']:.2f})")
            
            if asset_signals:
                signals_report[asset] = asset_signals
        
        return self.format_signals_report(signals_report)
    
    def format_signals_report(self, signals_report):
        """Format signals into readable report"""
        print("\n" + "="*60)
        print("📊 QUOTEX SIGNALS SUMMARY")
        print("="*60)
        
        strong_signals = []
        medium_signals = []
        
        for asset, timeframes in signals_report.items():
            for timeframe, signal_data in timeframes.items():
                signal_info = {
                    'asset': asset,
                    'timeframe': timeframe,
                    'signal': signal_data['signal'],
                    'confidence': signal_data['confidence'],
                    'price': signal_data['current_price'],
                    'trend': signal_data['trend']
                }
                
                if signal_data['confidence'] > 0.75:
                    strong_signals.append(signal_info)
                elif signal_data['confidence'] > 0.6:
                    medium_signals.append(signal_info)
        
        # Display strong signals first
        if strong_signals:
            print("\n🔥 STRONG SIGNALS (High Confidence):")
            for signal in strong_signals:
                print(f"   ✅ {signal['asset']} | {signal['timeframe']} | "
                      f"{signal['signal']} | Confidence: {signal['confidence']:.2f} | "
                      f"Price: ${signal['price']:.2f}")
        
        if medium_signals:
            print("\n🟡 MEDIUM SIGNALS (Good Confidence):")
            for signal in medium_signals:
                print(f"   ⚠️  {signal['asset']} | {signal['timeframe']} | "
                      f"{signal['signal']} | Confidence: {signal['confidence']:.2f} | "
                      f"Price: ${signal['price']:.2f}")
        
        if not strong_signals and not medium_signals:
            print("\n🟢 No high-confidence signals at the moment. "
                  "Wait for better market conditions.")
        
        return {
            'strong_signals': strong_signals,
            'medium_signals': medium_signals,
            'timestamp': datetime.now(),
            'total_signals': len(strong_signals) + len(medium_signals)
        }

# Usage example
if __name__ == "__main__":
    signal_gen = QuotexSignalGenerator()
    signals = signal_gen.generate_quotex_signals()