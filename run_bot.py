# run_bot.py
import os
import sys
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def show_welcome():
    print("🤖" * 20)
    print("🎯 QUOTEX AI TRADING BOT")
    print("🚀 Professional Signal Generator")
    print("🤖" * 20)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def main():
    show_welcome()
    
    while True:
        print("📋 MAIN MENU:")
        print("=" * 40)
        print("1. 🎯 Generate Signals Now")
        print("2. 🤖 Start Auto Bot (30min intervals)") 
        print("3. 📊 View Dashboard & Analytics")
        print("4. ⚡ Quick Test (Fast Analysis)")
        print("5. 📝 View Logs & Reports")
        print("6. ❌ Exit")
        print("=" * 40)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            print("\n🎯 GENERATING TRADING SIGNALS...")
            print("⏳ This may take 1-2 minutes...")
            try:
                from src.core.signal_generator import QuotexSignalGenerator
                generator = QuotexSignalGenerator()
                signals = generator.generate_quotex_signals()
                print(f"✅ Generated {signals['total_signals']} signals!")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
        elif choice == "2":
            print("\n🤖 STARTING AUTO TRADING BOT...")
            print("💡 Bot will run every 30 minutes")
            print("⏸️  Press Ctrl+C to stop")
            try:
                from src.core.auto_signal_bot import AutoSignalBot
                bot = AutoSignalBot()
                bot.start_auto_mode(interval_minutes=30)
                
            except KeyboardInterrupt:
                print("\n🛑 Auto bot stopped by user")
            except Exception as e:
                print(f"❌ Error: {e}")
            
        elif choice == "3":
            print("\n📊 LOADING DASHBOARD...")
            try:
                from src.core.dashboard import SignalDashboard
                dashboard = SignalDashboard()
                dashboard.show_basic_dashboard()
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
        elif choice == "4":
            print("\n⚡ QUICK TEST MODE...")
            print("🔄 Testing with limited assets for speed...")
            try:
                from src.core.signal_generator import QuotexSignalGenerator
                generator = QuotexSignalGenerator()
                
                # Test with only 2 assets for speed
                test_assets = ['BTC-USD', 'EURUSD=X']
                print(f"🔍 Analyzing: {', '.join(test_assets)}")
                
                # Override assets for quick test
                generator.asset_groups = {'quick_test': test_assets}
                signals = generator.generate_quotex_signals()
                print(f"✅ Quick test completed! Found {signals['total_signals']} signals")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
        elif choice == "5":
            print("\n📝 LOGS & REPORTS:")
            print("-" * 30)
            try:
                # Show recent log files
                import glob
                log_files = glob.glob("logs/*.txt") + glob.glob("logs/*.log")
                log_files.sort(reverse=True)
                
                if log_files:
                    print("Recent log files:")
                    for file in log_files[:5]:
                        file_size = os.path.getsize(file)
                        mod_time = datetime.fromtimestamp(os.path.getmtime(file))
                        print(f"   📄 {os.path.basename(file)}")
                        print(f"      Size: {file_size} bytes | Modified: {mod_time.strftime('%H:%M')}")
                else:
                    print("No log files found yet.")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
            
        elif choice == "6":
            print("\n👋 Thank you for using Quotex AI Trading Bot!")
            print("📈 Happy Trading! 🚀")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-6.")
        
        # Wait for user to continue
        if choice != "2":  # Don't wait if auto bot is running
            input("\nPress Enter to continue...")
        print("\n" + "="*50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Program stopped by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
