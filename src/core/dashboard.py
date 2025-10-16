# src/core/dashboard.py
import os
import glob
from datetime import datetime, timedelta
import sys

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

class SignalDashboard:
    def __init__(self):
        self.logs_dir = 'logs'
        print("✅ Dashboard initialized")
        
    def check_logs_directory(self):
        """Check if logs directory exists and has files"""
        print(f"\n🔍 Checking logs directory: {self.logs_dir}")
        
        if not os.path.exists(self.logs_dir):
            print("❌ Logs directory does not exist!")
            return False
        
        files = os.listdir(self.logs_dir)
        print(f"📁 Files in logs directory: {len(files)}")
        
        for file in files:
            print(f"   📄 {file}")
        
        return len(files) > 0
    
    def create_sample_signal(self):
        """Create a sample signal file for testing"""
        sample_file = f"logs/signals_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        
        try:
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write("QUOTEX TRADING SIGNALS REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Signals: 3\n\n")
                
                f.write("STRONG SIGNALS:\n")
                f.write("-" * 30 + "\n")
                f.write("✅ BTC-USD | 5m | STRONG_BUY | Conf: 0.85 | Price: $65200.50\n")
                f.write("✅ EURUSD=X | 15m | BUY | Conf: 0.78 | Price: $1.0852\n\n")
                
                f.write("MEDIUM SIGNALS:\n")
                f.write("-" * 30 + "\n")
                f.write("⚠️ ETH-USD | 1m | BUY | Conf: 0.68 | Price: $3520.75\n")
            
            print(f"✅ Sample signal file created: {sample_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating sample: {e}")
            return False
    
    def show_basic_dashboard(self):
        """Show basic dashboard information"""
        print("\n" + "="*60)
        print("📊 BASIC DASHBOARD")
        print("="*60)
        
        # Check logs directory
        has_logs = self.check_logs_directory()
        
        if not has_logs:
            print("\n🔄 Creating sample signals for testing...")
            self.create_sample_signal()
        
        # Show recent signals
        self.show_recent_signals_simple()
        
        # Show system info
        self.show_system_info()
    
    def show_recent_signals_simple(self):
        """Show recent signals in simple format"""
        print(f"\n📈 RECENT SIGNALS:")
        print("-" * 40)
        
        signal_files = glob.glob(os.path.join(self.logs_dir, "signals_*.txt"))
        signal_files.sort(reverse=True)
        
        if not signal_files:
            print("   No signal files found")
            print("   💡 Run 'python src/core/signal_generator.py' to generate signals")
            return
        
        for file_path in signal_files[:3]:  # Show last 3 files
            filename = os.path.basename(file_path)
            print(f"   📄 {filename}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[:6]:  # Show first 6 lines
                        if line.strip():
                            print(f"     {line.strip()}")
                print("")  # Empty line between files
                
            except Exception as e:
                print(f"     ❌ Error reading file: {e}")
    
    def show_system_info(self):
        """Show system information"""
        print(f"\n🔧 SYSTEM INFORMATION:")
        print("-" * 30)
        print(f"   🐍 Python: {sys.version.split()[0]}")
        print(f"   📁 Working Directory: {os.getcwd()}")
        print(f"   🕒 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check if required packages are installed
        try:
            import pandas
            print(f"   ✅ pandas: {pandas.__version__}")
        except ImportError:
            print(f"   ❌ pandas: Not installed")
        
        try:
            import yfinance
            print(f"   ✅ yfinance: Installed")
        except ImportError:
            print(f"   ❌ yfinance: Not installed")
        
        print(f"\n🎯 QUICK ACTIONS:")
        print("   1. Generate signals: python src/core/signal_generator.py")
        print("   2. Auto mode: python src/core/auto_signal_bot.py")
        print("   3. View logs: ls -la logs/")

if __name__ == "__main__":
    print("🚀 Starting Dashboard...")
    dashboard = SignalDashboard()
    dashboard.show_basic_dashboard()