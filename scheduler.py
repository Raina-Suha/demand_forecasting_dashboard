# scheduler.py
import schedule
import time
import os

def run_forecast():
    os.system("python forecast_script.py")

# Run every Monday at 8 AM
schedule.every().monday.at("08:00").do(run_forecast)

while True:
    schedule.run_pending()
    time.sleep(60)
