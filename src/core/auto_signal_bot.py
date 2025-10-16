# src/core/auto_signal_bot.py
import time
import schedule
from datetime import datetime
import logging
import os
import sys

# Fix import path - add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

try:
    from src.core.signal_generator import QuotexSignalGenerator
    print("✅ Successfully imported signal generator")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Trying alternative import...")
    # Alternative import
    from signal_generator import QuotexSignalGenerator

class AutoSignalBot:
    def __init__(self):
        self.generator = QuotexSignalGenerator()
        self.setup_logging()
        self.signal_count = 0
        
    def setup_logging(self):
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/auto_signals.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_signals(self):
        """Generate and log trading signals"""
        try:
            self.signal_count += 1
            print(f"\n{'='*60}")
            print(f"🔄 SIGNAL GENERATION #{self.signal_count}")
            print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            
            # Generate signals
            signals_report = self.generator.generate_quotex_signals()
            
            # Log results
            self.logger.info(f"Generated {signals_report['total_signals']} signals "
                           f"({len(signals_report['strong_signals'])} strong, "
                           f"{len(signals_report['medium_signals'])} medium)")
            
            # Save detailed report
            self.save_detailed_report(signals_report)
            
            return signals_report
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return None
    
    def save_detailed_report(self, signals_report):
        """Save detailed signals report to file"""
        try:
            report_file = f"logs/signals_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("QUOTEX TRADING SIGNALS REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Signals: {signals_report['total_signals']}\n\n")
                
                if signals_report['strong_signals']:
                    f.write("STRONG SIGNALS:\n")
                    f.write("-" * 30 + "\n")
                    for signal in signals_report['strong_signals']:
                        f.write(f"✅ {signal['asset']} | {signal['timeframe']} | "
                               f"{signal['signal']} | Conf: {signal['confidence']:.2f} | "
                               f"Price: ${signal['price']:.2f}\n")
                    f.write("\n")
                
                if signals_report['medium_signals']:
                    f.write("MEDIUM SIGNALS:\n")
                    f.write("-" * 30 + "\n")
                    for signal in signals_report['medium_signals']:
                        f.write(f"⚠️  {signal['asset']} | {signal['timeframe']} | "
                               f"{signal['signal']} | Conf: {signal['confidence']:.2f} | "
                               f"Price: ${signal['price']:.2f}\n")
            
            print(f"💾 Report saved: {report_file}")
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def start_auto_mode(self, interval_minutes=30):
        """Start automatic signal generation"""
        print("🤖 STARTING AUTO SIGNAL BOT")
        print(f"⏰ Interval: Every {interval_minutes} minutes")
        print(f"📁 Logs: ./logs/auto_signals.log")
        print("⏸️  Press Ctrl+C to stop\n")
        
        # Generate first signal immediately
        self.generate_signals()
        
        # Schedule subsequent signals
        schedule.every(interval_minutes).minutes.do(self.generate_signals)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print(f"\n🛑 Auto bot stopped. Generated {self.signal_count} signal sets.")
    
    def start_test_mode(self, iterations=3, interval_minutes=2):
        """Test mode with shorter intervals"""
        print("🧪 STARTING TEST MODE")
        print(f"🔄 Iterations: {iterations}")
        print(f"⏰ Interval: {interval_minutes} minutes")
        
        for i in range(iterations):
            print(f"\n🎯 TEST ITERATION {i+1}/{iterations}")
            self.generate_signals()
            
            if i < iterations - 1:
                print(f"\n⏳ Waiting {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

if __name__ == "__main__":
    bot = AutoSignalBot()
    
    # Ask user for mode
    print("Select mode:")
    print("1. Test Mode (3 iterations, 2 min intervals)")
    print("2. Auto Mode (continuous, 30 min intervals)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        bot.start_test_mode()
    else:
        bot.start_auto_mode()