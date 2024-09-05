import backtrader as bt
import yfinance as yf
from flask import Flask, render_template, request
import io
import base64

app = Flask(__name__)

class BuyAndHold(bt.Strategy):
    def __init__(self):
        pass

    def next(self):
        pass

class MovingAverageCrossover(bt.Strategy):
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SimpleMovingAverage(self.data, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SimpleMovingAverage(self.data, period=self.params.slow_period)

    def next(self):
        if self.fast_ma[0] > self.slow_ma[0] and self.fast_ma[-1] <= self.slow_ma[-1]:
            self.buy()
        elif self.fast_ma[0] < self.slow_ma[0] and self.fast_ma[-1] >= self.slow_ma[-1]:
            self.sell()

class RSI(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_upper', 70),
        ('rsi_lower', 30),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data, period=self.params.rsi_period)

    def next(self):
        if self.rsi[0] < self.params.rsi_lower:
            self.buy()
        elif self.rsi[0] > self.params.rsi_upper:
            self.sell()

class MACD(bt.Strategy):
    params = (
        ('macd_period1', 12),
        ('macd_period2', 26),
        ('signal_period', 9),
    )

    def __init__(self):
        self.macd = bt.indicators.MACDHisto(self.data, period_me1=self.params.macd_period1,
                                             period_me2=self.params.macd_period2, period_signal=self.params.signal_period)

    def next(self):
        if self.macd.lines.histo[0] > 0 and self.macd.lines.histo[-1] <= 0:
            self.buy()
        elif self.macd.lines.histo[0] < 0 and self.macd.lines.histo[-1] >= 0:
            self.sell()

STRATEGIES = {
    'Buy and Hold': BuyAndHold,
    'Moving Average Crossover': MovingAverageCrossover,
    'RSI Strategy': RSI,
    'MACD Strategy': MACD,
}

def get_strategy(strategy_name):
    return STRATEGIES.get(strategy_name, BuyAndHold)

@app.route('/')
def index():
    return render_template('index.html', strategies=STRATEGIES.keys())

@app.route('/plot', methods=['POST'])
def plot_results():
    # Retrieve user input from the form
    stock_symbol = request.form['stock_symbol']
    cash_amount = float(request.form['cash_amount'])
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    strategy_name = request.form['strategy']

    # Download data based on user input
    df = yf.download(stock_symbol, start=start_date, end=end_date)

    # Create a data feed
    feed = bt.feeds.PandasData(dataname=df)

    # Create cerebro
    cerebro = bt.Cerebro()

    # Add data feed to cerebro
    cerebro.adddata(feed)

    # Set initial cash for the broker
    cerebro.broker.setcash(cash_amount)

    # Get selected strategy
    strategy_class = get_strategy(strategy_name)
    cerebro.addstrategy(strategy_class)

    # Run the strategy
    cerebro.run()

    # Generate the plot using Matplotlib
    fig = cerebro.plot(style='candlestick', iplot=False, returnfig=True)

    # Convert the plot to a base64 encoded string
    img_str = plot_to_base64(fig)

    return render_template('plot.html', img_str=img_str)

def plot_to_base64(fig):
    # Save the plot to a BytesIO object
    img_bytes_io = io.BytesIO()
    fig.savefig(img_bytes_io, format='png')
    img_bytes_io.seek(0)

    # Encode the plot image as base64
    img_base64 = base64.b64encode(img_bytes_io.getvalue()).decode()
    return img_base64

if __name__ == '__main__':
    app.run(debug=True)
