#AI 602 Programming in Python
# Team 11

# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib.dates as mdates

# ===============================
# 2. CLASS-BASED DESIGN
# ===============================
class BitcoinStockAnalysis:
    def __init__(self):
        self.end_date = datetime.date.today()
        self.start_date_30 = self.end_date - datetime.timedelta(days=30)
        self.start_date_year = self.end_date - datetime.timedelta(days=365)
        self.download_data()

    def safe_download(self, ticker, **kwargs):
        for attempt in range(3):
            try:
                print(f"\n Downloading {ticker} (Attempt {attempt + 1})...")
                data = yf.download(ticker, **kwargs)
                if not data.empty:
                    return data
            except Exception as e:
                print(f" Attempt {attempt + 1} failed: {e}")
        print(f" Failed to download {ticker} after 3 attempts.")
        return pd.DataFrame()

    def download_data(self):
        self.btc_data_30d = self.safe_download('BTC-USD', start=self.start_date_30, end=self.end_date, auto_adjust=False)
        self.btc_data_full = self.safe_download('BTC-USD', start='2023-05-09', end=self.end_date, auto_adjust=False)
        self.aapl_data_full = self.safe_download('AAPL', start='2023-05-09', end=self.end_date)

        
        self.aapl_data_recent = self.aapl_data_full.copy()

        for df in [self.btc_data_30d, self.btc_data_full, self.aapl_data_full, self.aapl_data_recent]:
            df.reset_index(inplace=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]


    def explore_data(self, df, name):
        print(f"\n>>> {name} Data Sample:")
        print(df.head())
        print(f"\n>>> {name} Data Description:")
        print(df.describe())
        print(f"\n>>> {name} Data Info:")
        print(df.info())

    def show_basic_exploration(self):
        self.explore_data(self.btc_data_30d, "BTC (30 Days)")
        print("\n>>> Correlation with BTC Close Price:")
        print(self.btc_data_30d.corr()['Close'].sort_values(ascending=False))
        print("\n>>> Total Days:", len(self.btc_data_30d))
        print(">>> Total Fields:", len(self.btc_data_30d.columns))
        self.explore_data(self.aapl_data_full, "AAPL")

    def plot_btc_ohlc_chart(self):
        fig = go.Figure(data=go.Ohlc(
            x=self.btc_data_full['Date'],
            open=self.btc_data_full['Open'],
            high=self.btc_data_full['High'],
            low=self.btc_data_full['Low'],
            close=self.btc_data_full['Close']
        ))
        fig.update_layout(
            title="Bitcoin Price Chart (OHLC Chart)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False
        )
        fig.show()

    def plot_intraday_btc(self):
        print("\n Fetching intraday BTC data for today...")
        today = datetime.datetime.now().date()

        intraday_data = yf.download('BTC-USD',
                                    start=today,
                                    end=today + datetime.timedelta(days=1),
                                    interval='5m',
                                    progress=False)

        if intraday_data.empty:
            print(" No intraday data available — market might be closed or data is delayed.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(intraday_data.index, intraday_data['Close'], label='BTC Price (5min)', marker='o', color='orange')
        plt.title("BTC Intraday Price – Today (5-minute Interval)")
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_intraday_aapl(self):
        print("\n Fetching intraday AAPL data for today...")
        today = datetime.datetime.now().date()

        intraday_data = yf.download('AAPL',
                                    start=today,
                                    end=today + datetime.timedelta(days=1),
                                    interval='5m',
                                    progress=False)

        if intraday_data.empty:
            print(" No intraday data available — market might be closed or data is delayed.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(intraday_data.index, intraday_data['Close'], label='AAPL Price (5min)', marker='o', color='green')
        plt.title("AAPL Intraday Price – Today (5-minute Interval)")
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    

    def plot_btc_aapl_candlestick_comparison(self):
         print("\n Generating BTC vs AAPL Candlestick Comparison (Past 30 Days)...")


         btc = self.btc_data_30d.copy()
         aapl = self.aapl_data_recent.copy()

          # Keep only last 30 days of AAPL to match BTC
         aapl = aapl[pd.to_datetime(aapl['Date']) >= pd.to_datetime(self.start_date_30)]


         if btc.empty or aapl.empty:
             print(" Not enough data to generate comparison.")
             return

         fig = go.Figure()

          # BTC Candlestick
         fig.add_trace(go.Candlestick(
             x=btc['Date'],
             open=btc['Open'],
             high=btc['High'],
             low=btc['Low'],
             close=btc['Close'],
             name='BTC',
             increasing_line_color='green',
             decreasing_line_color='red'
          ))

          # AAPL Candlestick
         fig.add_trace(go.Candlestick(
             x=aapl['Date'],
             open=aapl['Open'],
             high=aapl['High'],
             low=aapl['Low'],
             close=aapl['Close'],
             name='AAPL',
             increasing_line_color='blue',
             decreasing_line_color='orange',
             opacity=0.5
         ))

         fig.update_layout(
             title="BTC vs AAPL – Past 30 Days (Candlestick Chart)",
             xaxis_title="Date",
             yaxis_title="Price (USD)",
             xaxis_rangeslider_visible=False,
             template="plotly_white",
             width=1200,
             height=600
         )

         fig.show()

    def plot_btc_open_close(self):
        data = self.btc_data_full.copy()
        data['Day'] = data['Date'].dt.strftime('%Y-%m-%d')
        data = data.iloc[::5]
        data = data.dropna(subset=['Open', 'Close'])

        data_long = pd.DataFrame({
            'Day': list(data['Day']) * 2,
            'Price Type': ['Open'] * len(data) + ['Close'] * len(data),
            'Value': list(data['Open']) + list(data['Close'])
        })

        fig = px.bar(
            data_long, x='Day', y='Value', color='Price Type', barmode='group',
            title='Bitcoin Open vs Close Prices (Column Chart)',
            color_discrete_map={'Open': 'crimson', 'Close': 'lightsalmon'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        fig.show()

    def compare_btc_windows(self):
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=30)

        btc_data = self.safe_download('BTC-USD', start=start_date, end=today)
        btc_data.reset_index(inplace=True)

        if btc_data.empty or len(btc_data) < 15:
            print(" Not enough data for forecasting.")
            return

        try:
            forecast_days = int(input(" Enter number of future days to forecast (e.g., 15): "))
            if forecast_days <= 0:
                print(" Forecast days must be a positive number.")
                return
        except:
            print(" Invalid input. Please enter a number.")
            return

        btc_data['7MA'] = btc_data['Close'].rolling(window=7).mean()
        trend_slope = (btc_data['7MA'].iloc[-1] - btc_data['7MA'].iloc[-8]) / 7

        btc_data['Daily_Change'] = btc_data['Close'].diff()
        volatility = btc_data['Daily_Change'].rolling(window=7).std().iloc[-1]

        last_price = btc_data['Close'].iloc[-1]
        future_prices = []
        future_dates = []
        np.random.seed(42)

        for i in range(1, forecast_days + 1):
            noise = np.random.normal(loc=0.0, scale=volatility)
            next_price = last_price + trend_slope + noise
            future_prices.append(next_price)
            last_price = next_price
            future_dates.append(btc_data['Date'].iloc[-1] + datetime.timedelta(days=i))

        plt.figure(figsize=(12, 6))
        plt.plot(btc_data['Date'], btc_data['Close'], label='Past 30 Days (Real)', marker='o')
        plt.plot(future_dates, future_prices, label=f'Next {forecast_days} Days (Forecast)', linestyle='--', marker='x', color='orange')
        plt.title(f"BTC Price Forecast – {forecast_days} Days Ahead")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def compare_aapl_windows(self):
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=30)

        aapl_data = self.safe_download('AAPL', start=start_date, end=today)
        aapl_data.reset_index(inplace=True)

        if aapl_data.empty or len(aapl_data) < 15:
            print(" Not enough data for forecasting.")
            return

        try:
            forecast_days = int(input(" Enter number of future days to forecast for AAPL (e.g., 15): "))
            if forecast_days <= 0:
                print(" Forecast days must be a positive number.")
                return
        except:
            print(" Invalid input. Please enter a number.")
            return

        aapl_data['7MA'] = aapl_data['Close'].rolling(window=7).mean()
        trend_slope = (aapl_data['7MA'].iloc[-1] - aapl_data['7MA'].iloc[-8]) / 7

        aapl_data['Daily_Change'] = aapl_data['Close'].diff()
        volatility = aapl_data['Daily_Change'].rolling(window=7).std().iloc[-1]

        last_price = aapl_data['Close'].iloc[-1]
        future_prices = []
        future_dates = []
        np.random.seed(42)

        for i in range(1, forecast_days + 1):
            noise = np.random.normal(loc=0.0, scale=volatility)
            next_price = last_price + trend_slope + noise
            future_prices.append(next_price)
            last_price = next_price
            future_dates.append(aapl_data['Date'].iloc[-1] + datetime.timedelta(days=i))

        plt.figure(figsize=(12, 6))
        plt.plot(aapl_data['Date'], aapl_data['Close'], label='AAPL Past 30 Days', marker='o')
        plt.plot(future_dates, future_prices, label=f'AAPL Next {forecast_days} Days', linestyle='--', marker='x', color='blue')
        plt.title(f"AAPL Price Forecast – {forecast_days} Days Ahead")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_grouped_aapl(self):
        data = self.aapl_data_recent.copy()
        data = data.iloc[::5]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=data['Date'], y=data['Open'], name='Open Price', marker_color='darkblue'))
        fig.add_trace(go.Bar(x=data['Date'], y=data['Close'], name='Close Price', marker_color='darkred'))
        fig.add_trace(go.Bar(x=data['Date'], y=data['High'], name='High Price', marker_color='darkgreen'))
        fig.add_trace(go.Bar(x=data['Date'], y=data['Low'], name='Low Price', marker_color='indigo'))

        names = cycle(['AAPL Stock Open Price', 'AAPL Stock Close Price', 'AAPL Stock High Price', 'AAPL Stock Low Price'])
        for trace, name in zip(fig.data, names):
            trace.name = name

        fig.update_layout(
            barmode='group',
            title_text='AAPL Stock Daily (Grouped Bar Chart)',
            font_size=15,
            xaxis_tickangle=-45,
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            plot_bgcolor='rgba(0,0,0,0)',
            width=1400
        )
        fig.show()

# ===============================
# 3. DRIVER CODE
# ===============================
if __name__ == "__main__":
    analysis = BitcoinStockAnalysis()
    analysis.show_basic_exploration()
    analysis.plot_btc_ohlc_chart()
    analysis.plot_intraday_btc()
    analysis.plot_intraday_aapl()
    analysis.plot_btc_aapl_candlestick_comparison()
    analysis.plot_btc_open_close()
    analysis.compare_btc_windows()
    analysis.compare_aapl_windows()
    analysis.plot_grouped_aapl()

