from django.shortcuts import render
from django.http import HttpResponse
from django.utils.safestring import mark_safe

import yfinance as yf
from datetime import datetime
import requests
import warnings

warnings.filterwarnings('ignore')



# Create your views here.
def index(request):
    #return HttpResponse("This is homepage")
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def prediction(request):
    return render(request,'prediction.html')

def linearprediction(request):
    if request.method == "POST":
        # Get the form data
        stock_name = request.POST.get('stock_name')
        stock_date = request.POST.get('stock_date')
        print(stock_name, stock_date)

        # Fetch stock data
        raw_data = fetch_stock_data(stock_name)
        print(raw_data)



        # preprocess the data
        preprocessed_data = get_preprocessed_data(raw_data)
        print(preprocessed_data)



        # Run regression analysis
        data,y_test,y_pred,future_close_prices = run_regression_analysis(preprocessed_data)

        last_known_close = data['Close'].iloc[-1]
        predicted_future_close = future_close_prices[-1]

        print("Last known close price:", last_known_close)
        print("future close prices:", future_close_prices)


        #this is for charts display
        data_visualization(data,y_test,y_pred,future_close_prices,stock_date)


        if predicted_future_close > last_known_close + (0.05 * last_known_close):
            output = {
            "stock_name" : stock_name,
            "current_price" : str(round(last_known_close,2)),
            "output" : "The stock is predicted to be <span style='color: #00E400;'>BULLISH</span> with predicted price after 7 days as : " + str(round(predicted_future_close,2))
            }
        elif predicted_future_close > last_known_close + (0.01 * last_known_close):
            output = {
            "stock_name" : stock_name,
            "current_price" : str(round(last_known_close,2)),
            "output" : mark_safe("The stock is predicted to be <span style='color: #00E400;'>SLIGHTLY BULLISH</span> with predicted price after 7 days as : " + str(round(predicted_future_close,2)))
            }
        elif predicted_future_close < last_known_close - (0.05 * last_known_close):
            output = {
            "stock_name" : stock_name,
            "current_price" : str(round(last_known_close,2)),
            "output" : mark_safe("The stock is predicted to be <span style='color: #FF0000;'>BEARISH</span> with predicted price after 7 days as : " + str(round(predicted_future_close,2)))
            }
        elif predicted_future_close < last_known_close - (0.01 * last_known_close):
            output = {
            "stock_name" : stock_name,
            "current_price" : str(round(last_known_close,2)),
            "output" : mark_safe("The stock is predicted to be <span style='color: #FF0000;'>SLIGHTLY BEARISH</span> with predicted price after 7 days as : " + str(round(predicted_future_close,2)))
            }
        else:
            output = {
            "stock_name" : stock_name,
            "current_price" : str(round(last_known_close,2)),
            "output" : mark_safe("The stock is predicted to be <span style='color: #0000FF;'>SIDEWAYS</span> with predicted price after 7 days as : " + str(round(predicted_future_close,2)))    
            }

        print("reached here 3")
        return render(request, 'linearprediction.html', output)

    else:
        return render(request, 'linearprediction.html')



def visualization(request):
    if request.method == "POST":
        # Get the form data
        stock_name = request.POST.get('stock_name')
        stock_date = request.POST.get('stock_date')
        chart_type = request.POST.get('chart_type')

        print(stock_name, stock_date)

        # Fetch stock data
        raw_data = fetch_stock_data(stock_name)
        print(raw_data)

        if chart_type == "candle":
            only_visualization_candle(raw_data,stock_date)
            return render(request, 'onlyVisCandle.html')
        
        elif chart_type == "line":
            only_visualization_line(raw_data,stock_date)
            return render(request, 'onlyVisLine.html')
        
        elif chart_type == "OHLC":
            only_visualization_OHLC(raw_data,stock_date)
            return render(request,'onlyVisOHLC.html')
        


        # if any problem then default this will render
        return render(request, 'onlyVisCandle.html')
    else:
        return render(request,'visualization.html')


def fullCandle(request):
    stock_name = request.GET.get('stock_name')
    stock_date = request.GET.get('stock_date')

    raw_data = fetch_stock_data(stock_name)

    print(stock_name, stock_date)
    print(raw_data)

    only_visualization_candle(raw_data,stock_date)
    return render(request, 'onlyVisCandle.html')


def latestnews(request):
    return render(request, 'latestnews.html')

def watchlist(request):
    return render(request, 'watchlist.html')







# Fetch stock data

global_stock = None

def fetch_stock_data(stock_name):
    try:
        global global_stock
        stock = stock_name
        
        info = yf.Ticker(stock).info
        
        try:
            unix_time = info['firstTradeDateEpochUtc']
        except KeyError:
            print("the name is incorrect")
            exit()
            
        # Convert Unix time to datetime
        dt = datetime.fromtimestamp(unix_time)

        # Format datetime to date
        date = dt.strftime('%Y-%m-%d')
        
        # Get today's date
        end_date = datetime.today().strftime('%Y-%m-%d')

        # Download stock data from the first trade date to today
        try:
            data = yf.download(stock, start=date, end=end_date)
        except ValueError:
            print("Failed to fetch data. Please try again later.")
            exit()
            
        
        if data.empty:
            print("no data available for stock")
            exit()
        
        global_stock = stock
        
    except requests.exceptions.ConnectionError:
        print("No Internet Connection")
    
    return data


# preprocessing the data

def get_preprocessed_data(raw_data):
    # Fetch the stock data
    data = raw_data

    # Calculate the 50-day moving average of the closing prices
    data['50_day_ma'] = data['Close'].rolling(window=50).mean()

    # Calculate the 200-day moving average of the closing prices
    data['200_day_ma'] = data['Close'].rolling(window=200).mean()

    # Fill NaN values with the mean of the available data
    data['50_day_ma'] = data['50_day_ma'].fillna(data['50_day_ma'].mean())
    data['200_day_ma'] = data['200_day_ma'].fillna(data['200_day_ma'].mean())

    return data




#  regression model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

next_day_open2=None

def run_regression_analysis(preprocessed_data):
    global next_day_open2
    data = preprocessed_data

    if data is None:
        print("Error: Preprocessed data could not be loaded.")
        return

    X = data[['Open']]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Model coefficient:", model.coef_)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error on test set:", mse)
    print("Mean Absolute Error on test set:", mae)
    print("R-squared on test set:", r2)

    future_close_prices = [data['Close'].iloc[-1]]
    next_day_open2=[data['Close'].iloc[-1]]

    for x in range(7):
        next_day_open = future_close_prices[-1]
        if x!=0:
            next_day_open2.append(future_close_prices[-1])
        X_future = np.array(next_day_open).reshape(-1, 1)
        next_day_close = model.predict(X_future)
        future_close_prices.append(next_day_close[0])

    return data,y_test,y_pred,future_close_prices[1:]





import warnings
import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def data_visualization(data,y_test,y_pred,future_close_prices,fetched_stock_date):
    start_date = pd.to_datetime(fetched_stock_date)

    data = data[data.index >= start_date]

    ap1 = mpf.make_addplot(data['50_day_ma'])
    ap2 = mpf.make_addplot(data['200_day_ma'])

    mpf.plot(data, type='candle', style='binance',title='Candlestick chart of ' + global_stock,
             ylabel='Price', volume=True, addplot=[ap1, ap2],savefig='static/candle.png')
    
    plt.figure()

    y_test_last_30 = y_test[-30:]
    y_pred_last_30 = y_pred[-30:]

    x = np.arange(len(y_test_last_30))

    plt.bar(x - 0.2, y_test_last_30, 0.4, label='actual')
    plt.bar(x + 0.2, y_pred_last_30, 0.4, label='predicted')

    plt.title('Comparison of Actual and Predicted Values')
    
    plt.legend()
    
    plt.savefig('static/comparison.png')

    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)

    plt.figure(figsize=(10, 6))
    plt.plot(future_dates, future_close_prices, marker='o', linestyle='-')

    plt.xlabel('Date')
    plt.ylabel('Predicted Close Price')
    plt.title('Predicted Close Prices for the Next 7 Days')

    plt.gcf().autofmt_xdate()
    
    plt.savefig('static/prediction.png')


import plotly.graph_objects as go
def only_visualization_candle(data,fetched_stock_date):
    # start_date = pd.to_datetime(fetched_stock_date)

    # data = data[data.index >= start_date]

    # mpf.plot(data, type='candle', style='binance',title='Candlestick chart of ' + global_stock,
    #          ylabel='Price', volume=True,savefig='static/onlyVisCandle.png')

    start_date = pd.to_datetime(fetched_stock_date)
    data = data[data.index >= start_date]

    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])

    fig.update_layout(title='Interactive Candlestick chart of ' + global_stock,
                      yaxis_title='Price')

    fig.write_html('templates/onlyVisCandle.html')




def only_visualization_line(data,fetched_stock_date):
    start_date = pd.to_datetime(fetched_stock_date)
    data = data[data.index >= start_date]

    fig = go.Figure(data=[go.Scatter(x=data.index,
                                     y=data['Close'],
                                     mode='lines+markers',
                                     hovertemplate=
                                     'Date: %{x}<br>' +
                                     'Close Price: %{y}<br>',
                                     name='')])

    fig.update_layout(title='Interactive Line chart of ' + global_stock,
                      xaxis_title='Date',yaxis_title='Price')

    fig.write_html('templates/onlyVisLine.html')





def only_visualization_OHLC(data,fetched_stock_date):
    start_date = pd.to_datetime(fetched_stock_date)
    data = data[data.index >= start_date]

    fig  = go.Figure(data=[go.Ohlc(x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'])])

    fig.update_layout(title='Interactive Line chart of ' + global_stock,
                      xaxis_title='Date',yaxis_title='Price')

    fig.write_html('templates/onlyVisOHLC.html')