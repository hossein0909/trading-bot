import pandas as pd
import yfinance as yf
import numpy as np
import time
import logging
import warnings
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
import json

warnings.filterwarnings('ignore')

print("🚀 ULTRA ADVANCED AI TRADING BOT STARTING...")

class UltraAdvancedTradingBot:
    def __init__(self):
        self.setup_logging()
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.positions = []
        self.trade_history = []
        self.is_running = False
        self.performance_metrics = {}
        self.risk_manager = RiskManager()
        self.market_analyzer = MarketAnalyzer()
        self.strategy_engine = StrategyEngine()
        
        print("✅ Ultra Advanced AI Bot Initialized")
        print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/trading_bot.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_multiple_market_data(self):
        """Get data for multiple assets"""
        assets = ['BTC-USD', 'ETH-USD', 'AAPL', 'SPY']
        market_data = {}
        
        for asset in assets:
            try:
                ticker = yf.Ticker(asset)
                data = ticker.history(period='5d', interval='5m')
                
                if not data.empty:
                    analyzed_data = self.market_analyzer.calculate_all_indicators(data)
                    market_data[asset] = {
                        'current_price': float(data['Close'].iloc[-1]),
                        'data': analyzed_data,
                        'trend': self.market_analyzer.analyze_trend(analyzed_data),
                        'volatility': self.market_analyzer.calculate_volatility(analyzed_data)
                    }
                    print(f"📊 {asset}: ${market_data[asset]['current_price']:.2f} - {market_data[asset]['trend']}")
                    
            except Exception as e:
                print(f"❌ Error fetching {asset}: {e}")
        
        return market_data
    
    def advanced_ai_analysis(self, market_data):
        """Advanced AI analysis with multiple strategies"""
        signals = {}
        
        for asset, data in market_data.items():
            # Multi-strategy analysis
            strategy_signals = self.strategy_engine.run_all_strategies(data['data'])  # FIX: pass data['data']
            
            # Risk assessment
            risk_score = self.risk_manager.assess_trade_risk(asset, data, strategy_signals)
            
            # Final decision
            if risk_score['can_trade'] and strategy_signals['combined_confidence'] > 0.7:
                signals[asset] = {
                    'signal': strategy_signals['primary_signal'],
                    'confidence': strategy_signals['combined_confidence'],
                    'risk_score': risk_score['score'],
                    'position_size': risk_score['recommended_size'],
                    'strategies_used': strategy_signals['active_strategies']
                }
        
        return signals
    
    def execute_advanced_trades(self, signals, market_data):
        """Execute trades with advanced risk management"""
        executed_trades = []
        
        for asset, signal_info in signals.items():
            if self.risk_manager.approve_trade(signal_info, self.balance):
                
                trade_details = {
                    'asset': asset,
                    'timestamp': datetime.now(),
                    'signal': signal_info['signal'],
                    'price': market_data[asset]['current_price'],
                    'size': signal_info['position_size'] * self.balance,  # FIX: calculate actual amount
                    'confidence': signal_info['confidence'],
                    'risk_score': signal_info['risk_score']
                }
                
                # Simulate trade execution
                self.balance -= trade_details['size']
                self.positions.append(trade_details)
                self.trade_history.append(trade_details)
                executed_trades.append(trade_details)
                
                print(f"🎯 TRADE EXECUTED: {asset} - {signal_info['signal']}")
                print(f"   💰 Size: ${trade_details['size']:.2f}")
                print(f"   📈 Price: ${trade_details['price']:.2f}")
                print(f"   💪 Confidence: {trade_details['confidence']:.2f}")
                print(f"   ⚠️  Risk Score: {trade_details['risk_score']:.2f}")
        
        return executed_trades
    
    def run_comprehensive_analysis(self):
        """Run complete market analysis"""
        print("\n" + "="*60)
        print("🔍 COMPREHENSIVE MARKET ANALYSIS")
        print("="*60)
        
        # 1. Get market data
        market_data = self.get_multiple_market_data()
        if not market_data:
            return None
        
        # 2. AI Analysis
        signals = self.advanced_ai_analysis(market_data)
        
        # 3. Market Regime Detection
        market_regime = self.market_analyzer.detect_market_regime(market_data)
        print(f"🏛️  Market Regime: {market_regime}")
        
        return {
            'market_data': market_data,
            'signals': signals,
            'market_regime': market_regime,
            'timestamp': datetime.now()
        }
    
    def start_advanced_trading(self, iterations=4):
        """Start advanced trading session"""
        print("🚀 STARTING ULTRA ADVANCED TRADING SESSION")
        print(f"🔄 Total Iterations: {iterations}")
        
        self.is_running = True
        session_trades = []
        
        for i in range(iterations):
            if not self.is_running:
                break
                
            print(f"\n{'#'*50}")
            print(f"📍 ITERATION {i+1}/{iterations}")
            print(f"{'#'*50}")
            
            try:
                # Comprehensive analysis
                analysis_result = self.run_comprehensive_analysis()
                
                if analysis_result and analysis_result['signals']:
                    # Execute trades
                    executed_trades = self.execute_advanced_trades(
                        analysis_result['signals'], 
                        analysis_result['market_data']
                    )
                    session_trades.extend(executed_trades)
                else:
                    print("🟡 No high-confidence signals found")
                
                # Performance update
                self.update_performance_metrics()
                
            except Exception as e:
                print(f"❌ Error in iteration {i+1}: {e}")
                self.logger.error(f"Iteration {i+1} error: {e}")
            
            # Wait between iterations
            if i < iterations - 1:
                print(f"\n⏳ Waiting 15 seconds for next analysis...")
                time.sleep(15)
        
        print("\n🎉 TRADING SESSION COMPLETED!")
        self.generate_comprehensive_report(session_trades)
    
    def update_performance_metrics(self):
        """Update real-time performance metrics"""
        total_trades = len(self.trade_history)
        if total_trades > 0:
            winning_trades = len([t for t in self.trade_history if t.get('profit', 0) > 0])
            win_rate = winning_trades / total_trades
            
            self.performance_metrics = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'current_balance': self.balance,
                'total_profit': self.balance - self.initial_balance
            }
    
    def generate_comprehensive_report(self, session_trades):
        """Generate detailed trading report"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE TRADING REPORT")
        print("="*60)
        
        print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
        print(f"💰 Current Balance: ${self.balance:,.2f}")
        print(f"📈 Net P/L: ${self.balance - self.initial_balance:,.2f}")
        
        if self.trade_history:
            print(f"\n🔢 Total Trades: {len(self.trade_history)}")
            print(f"🎯 Session Trades: {len(session_trades)}")
            
            # Risk metrics
            max_position = max([t['size'] for t in self.trade_history]) if self.trade_history else 0
            print(f"⚡ Max Position Size: ${max_position:.2f}")
            
        print(f"\n📈 Performance Metrics:")
        for metric, value in self.performance_metrics.items():
            print(f"   {metric.replace('_', ' ').title()}: {value}")

class RiskManager:
    def __init__(self):
        self.max_position_size = 0.1  # 10% of balance per trade
        self.max_daily_loss = 0.05    # 5% max daily loss
        self.risk_free_rate = 0.02    # 2% risk free rate
    
    def assess_trade_risk(self, asset, data, signals):
        """Assess risk for a potential trade"""
        volatility = data.get('volatility', 0.1)
        confidence = signals.get('combined_confidence', 0.5)
        
        # Calculate risk score (0-1, lower is better)
        risk_score = max(0.1, volatility * (1 - confidence))
        
        # Position sizing based on Kelly Criterion
        recommended_size = self.calculate_position_size(confidence, risk_score)
        
        return {
            'score': risk_score,
            'can_trade': risk_score < 0.3 and confidence > 0.6,
            'recommended_size': recommended_size
        }
    
    def calculate_position_size(self, confidence, risk_score):
        """Calculate optimal position size using Kelly Criterion"""
        kelly_fraction = confidence - (1 - confidence) / (1 / risk_score - 1)
        position_size = max(0.01, min(self.max_position_size, kelly_fraction * 0.5))  # Half Kelly
        return position_size
    
    def approve_trade(self, signal_info, current_balance):
        """Final approval for trade execution"""
        required_size = signal_info['position_size']
        return required_size <= current_balance * self.max_position_size

class MarketAnalyzer:
    def calculate_all_indicators(self, data):
        """Calculate all technical indicators"""
        if data.empty:
            return data
            
        df = data.copy()
        
        # Price-based indicators
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
        
        # Volume indicators
        df['volume_sma'] = df['Volume'].rolling(10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        return df.dropna()
    
    def analyze_trend(self, data):
        """Analyze market trend"""
        if data.empty:
            return "NEUTRAL"
            
        current_price = data['Close'].iloc[-1]
        sma_20 = data['sma_20'].iloc[-1]
        
        if current_price > sma_20 * 1.02:
            return "BULLISH"
        elif current_price < sma_20 * 0.98:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def calculate_volatility(self, data, window=20):
        """Calculate price volatility"""
        if len(data) < window:
            return 0.1
            
        returns = data['Close'].pct_change().dropna()
        volatility = returns.rolling(window=window).std().iloc[-1]
        return volatility if not np.isnan(volatility) else 0.1
    
    def detect_market_regime(self, market_data):
        """Detect current market regime"""
        total_assets = len(market_data)
        bullish_count = sum(1 for data in market_data.values() if data['trend'] == 'BULLISH')
        
        if bullish_count / total_assets > 0.7:
            return "STRONG_BULL"
        elif bullish_count / total_assets < 0.3:
            return "STRONG_BEAR"
        else:
            return "MIXED"

class StrategyEngine:
    def __init__(self):
        self.strategies = {
            'momentum': self.momentum_strategy,
            'mean_reversion': self.mean_reversion_strategy,
            'breakout': self.breakout_strategy,
            'rsi_oversold': self.rsi_oversold_strategy
        }
    
    def run_all_strategies(self, data):
        """Run all trading strategies"""
        signals = {}
        active_strategies = []
        total_confidence = 0
        strategy_count = 0
        
        for name, strategy in self.strategies.items():
            signal, confidence = strategy(data)
            if signal != "HOLD":
                signals[name] = {'signal': signal, 'confidence': confidence}
                active_strategies.append(name)
                total_confidence += confidence
                strategy_count += 1
        
        # Calculate combined signal
        primary_signal = self.aggregate_signals(signals)
        combined_confidence = total_confidence / max(strategy_count, 1)
        
        return {
            'primary_signal': primary_signal,
            'combined_confidence': combined_confidence,
            'active_strategies': active_strategies,
            'strategy_details': signals
        }
    
    def momentum_strategy(self, data):
        """Momentum-based strategy"""
        if data.empty:
            return "HOLD", 0.0
            
        price_trend = data['Close'].iloc[-1] > data['sma_20'].iloc[-1]
        macd_trend = data['macd'].iloc[-1] > data['macd_signal'].iloc[-1]
        
        if price_trend and macd_trend:
            return "BUY", 0.7
        elif not price_trend and not macd_trend:
            return "SELL", 0.7
        else:
            return "HOLD", 0.3
    
    def mean_reversion_strategy(self, data):
        """Mean reversion strategy"""
        if data.empty:
            return "HOLD", 0.0
            
        current_price = data['Close'].iloc[-1]
        bb_upper = data['bb_upper'].iloc[-1]
        bb_lower = data['bb_lower'].iloc[-1]
        
        if current_price > bb_upper:
            return "SELL", 0.8
        elif current_price < bb_lower:
            return "BUY", 0.8
        else:
            return "HOLD", 0.4
    
    def breakout_strategy(self, data):
        """Breakout strategy"""
        if len(data) < 2:
            return "HOLD", 0.0
            
        current_high = data['High'].iloc[-1]
        previous_high = data['High'].iloc[-2]
        
        if current_high > previous_high * 1.01:  # 1% breakout
            return "BUY", 0.75
        else:
            return "HOLD", 0.3
    
    def rsi_oversold_strategy(self, data):
        """RSI overbought/oversold strategy"""
        if data.empty:
            return "HOLD", 0.0
            
        rsi = data['rsi'].iloc[-1]
        
        if rsi < 30:
            return "BUY", 0.8
        elif rsi > 70:
            return "SELL", 0.8
        else:
            return "HOLD", 0.4
    
    def aggregate_signals(self, signals):
        """Aggregate signals from all strategies"""
        if not signals:
            return "HOLD"
            
        buy_signals = sum(1 for s in signals.values() if s['signal'] == 'BUY')
        sell_signals = sum(1 for s in signals.values() if s['signal'] == 'SELL')
        
        if buy_signals > sell_signals:
            return "BUY"
        elif sell_signals > buy_signals:
            return "SELL"
        else:
            return "HOLD"

if __name__ == "__main__":
    try:
        bot = UltraAdvancedTradingBot()
        bot.start_advanced_trading(iterations=4)
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"💥 Critical error: {e}")
        logging.error(f"Critical error: {e}")