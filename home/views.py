from django.shortcuts import render
from django.http import HttpResponse
from django.utils.safestring import mark_safe

import yfinance as yf
from datetime import datetime
import requests
import json
import warnings
from .forms import NotificationForm
from celery import shared_task
from django.core.mail import send_mail
from .models import Notification


from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.metrics import mean_squared_error, mean_absolute_error


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
    


def movingaverage(request):
    if request.method == "POST":
        # Get the form data
        stock_name = request.POST.get('stock_name')
        stock_date = request.POST.get('stock_date')
        print(stock_name, stock_date)

        # Fetch stock data
        raw_data = fetch_stock_data(stock_name)
        print(raw_data)



        # preprocess the data
        data = get_preprocessed_data(raw_data)
        print(data)

        data['ma_diff'] = data['50_day_ma'] - data['200_day_ma']

        crossings = np.where(np.diff(np.sign(data['ma_diff'])))[0]

        last_crossing = crossings[-1]

        last_known_close = data['Close'].iloc[-1]

        #6 dates 3bull 3bear for marker
        date_bull_1 = None
        date_bull_2 = None
        date_bull_3 = None

        date_bear_1 = None
        date_bear_2 = None
        date_bear_3 = None

        if data['ma_diff'].iloc[last_crossing+1] > 0:
            output = {
            "stock_name" : stock_name,
            "current_price" : str(round(last_known_close,2)),
            "output" : mark_safe("The stock started being <span style='color: #00E400;'>BULLISH</span> on: " + str(data.index[last_crossing+1]))
            }
            print('The stock started being bullish on', data.index[last_crossing+1]) 
            print('bull 1')
            date_bull_1 = data.index[last_crossing+1]
        else:
            output = {
            "stock_name" : stock_name,
            "current_price" : str(round(last_known_close,2)),
            "output" : mark_safe("The stock started being <span style='color: #FF0000;'>BEARISH</span> on: " + str(data.index[last_crossing+1]))
            }
            print('The stock started being bearish on', data.index[last_crossing+1])
            print('bear 1')
            date_bear_1 = data.index[last_crossing+1]

        ma_diff_diff = data['ma_diff'].iloc[last_crossing + 2:].diff()

        if data['ma_diff'].iloc[last_crossing+1] > 0:
            negative_days = 0
            for date, value in ma_diff_diff.items():
                if value < 0 and data['Close'].loc[date] < data['50_day_ma'].loc[date]:
                    negative_days += 1
                    if negative_days >= 7:
                        output = {
                        "stock_name" : stock_name,
                        "current_price" : str(round(last_known_close,2)),
                        "output" : mark_safe("The market has become <span style='color: #FF0000;'>BEARISH</span> on: " + str(date))
                        }
                        print('if wala bearish')
                        print('The market has become bearish on', date)
                        print('bear 2')
                        date_bear_2 = date
                        break
                else:
                    negative_days = 0
            positive_days = 0
            date_index = list(ma_diff_diff.keys()).index(date)

            for date, value in list(ma_diff_diff.items())[date_index:]:
                if value > 0 and data['Close'].loc[date] > data['50_day_ma'].loc[date]:
                    positive_days += 1
                    if positive_days >= 7:
                        output = {
                        "stock_name" : stock_name,
                        "current_price" : str(round(last_known_close,2)),
                        "output" : mark_safe("The market has become <span style='color: #00E400;'>BULLISH</span> on: " + str(date))
                        }
                        print('The market has become bullish on', date)
                        print('bull 2')
                        date_bull_2 = date
                        break
        else:
            positive_days = 0
            for date, value in ma_diff_diff.items():
                if value > 0 and data['Close'].loc[date] > data['50_day_ma'].loc[date]:
                    positive_days += 1
                    if positive_days >= 7:
                        output = {
                        "stock_name" : stock_name,
                        "current_price" : str(round(last_known_close,2)),
                        "output" : mark_safe("The market has become <span style='color: #00E400;'>BULLISH</span> on: " + str(date))
                        }
                        print('The market has become bullish on', date)
                        print('bull 3')
                        date_bull_3 = date
                        break
                else:
                    positive_days = 0
            negative_days = 0
            date_index = list(ma_diff_diff.keys()).index(date)

            for date, value in list(ma_diff_diff.items())[date_index:]:
                if value < 0 and data['Close'].loc[date] < data['50_day_ma'].loc[date]:
                    negative_days += 1
                    if negative_days >= 7:
                        output = {
                        "stock_name" : stock_name,
                        "current_price" : str(round(last_known_close,2)),
                        "output" : mark_safe("The market has become <span style='color: #FF0000;'>BEARISH</span> on: " + str(date))
                        }
                        print('else wala bearish')
                        print('The market has become bearish on', date)
                        print('bear 3')
                        date_bear_3 = date
                        break
        start_date = data.index[last_crossing] - pd.Timedelta(days=7)

        selected_data = data.loc[start_date:]

        ohlc = selected_data[['Open', 'High', 'Low', 'Close']]

        ma50_plot = mpf.make_addplot(selected_data['50_day_ma'], color='blue')
        ma200_plot = mpf.make_addplot(selected_data['200_day_ma'], color='red')

        fig, axes = mpf.plot(ohlc, type='candle', style='binance', title='Candlestick chart of ' + stock_name, ylabel='Price', addplot=[ma50_plot, ma200_plot], returnfig=True)

        print('all 6 dates check')
        print('date_bear_1 ' + str(date_bear_1 if date_bear_1 else 'None'))
        print('date_bear_2 ' + str(date_bear_2 if date_bear_2 else 'None'))
        print('date_bear_3 ' + str(date_bear_3 if date_bear_3 else 'None'))
        print('date_bull_1 ' + str(date_bull_1 if date_bull_1 else 'None'))
        print('date_bull_2 ' + str(date_bull_2 if date_bull_2 else 'None'))
        print('date_bull_3 ' + str(date_bull_3 if date_bull_3 else 'None'))

        if(date_bull_1):
            # Convert the date string to a pandas Timestamp
            x_date = pd.to_datetime(str(date_bull_1))

            # Get the index of the date
            x_index = ohlc.index.get_loc(x_date)

            # Convert the index to a fraction of the total length of the x-axis
            x_value = x_index / len(ohlc)
            print(x_value)

            # Now you can use x_value in your code
            axes[0].annotate('',
                    xy=(x_value, 0.5), 
                    xycoords='axes fraction', 
                    xytext=(x_value, 0.4), 
                    arrowprops=dict(facecolor='#00E400', shrink=0.05))
            
        
        if(date_bull_2):
            # Convert the date string to a pandas Timestamp
            x_date = pd.to_datetime(str(date_bull_2))

            # Get the index of the date
            x_index = ohlc.index.get_loc(x_date)

            # Convert the index to a fraction of the total length of the x-axis
            x_value = x_index / len(ohlc)
            print(x_value)

            # Now you can use x_value in your code
            axes[0].annotate('',
                    xy=(x_value, 0.5), 
                    xycoords='axes fraction', 
                    xytext=(x_value, 0.4), 
                    arrowprops=dict(facecolor='#00E400', shrink=0.05))
            
        if(date_bull_3):
            # Convert the date string to a pandas Timestamp
            x_date = pd.to_datetime(str(date_bull_3))

            # Get the index of the date
            x_index = ohlc.index.get_loc(x_date)

            # Convert the index to a fraction of the total length of the x-axis
            x_value = x_index / len(ohlc)
            print(x_value)

            # Now you can use x_value in your code
            axes[0].annotate('',
                    xy=(x_value, 0.5), 
                    xycoords='axes fraction', 
                    xytext=(x_value, 0.4), 
                    arrowprops=dict(facecolor='#00E400', shrink=0.05))
            
        
        if(date_bear_1):
            # Convert the date string to a pandas Timestamp
            x_date = pd.to_datetime(str(date_bear_1))

            # Get the index of the date
            x_index = ohlc.index.get_loc(x_date)

            # Convert the index to a fraction of the total length of the x-axis
            x_value = x_index / len(ohlc)
            print(x_value)

            # Now you can use x_value in your code
            axes[0].annotate('',
                    xy=(x_value, 0.5), 
                    xycoords='axes fraction', 
                    xytext=(x_value, 0.6), 
                    arrowprops=dict(facecolor='red', shrink=0.05))
        
        if(date_bear_2):
            # Convert the date string to a pandas Timestamp
            x_date = pd.to_datetime(str(date_bear_2))

            # Get the index of the date
            x_index = ohlc.index.get_loc(x_date)

            # Convert the index to a fraction of the total length of the x-axis
            x_value = x_index / len(ohlc)
            print(x_value)

            # Now you can use x_value in your code
            axes[0].annotate('',
                    xy=(x_value, 0.5), 
                    xycoords='axes fraction', 
                    xytext=(x_value, 0.6), 
                    arrowprops=dict(facecolor='red', shrink=0.05))
            
        if(date_bear_3):
            # Convert the date string to a pandas Timestamp
            x_date = pd.to_datetime(str(date_bear_3))

            # Get the index of the date
            x_index = ohlc.index.get_loc(x_date)

            # Convert the index to a fraction of the total length of the x-axis
            x_value = x_index / len(ohlc)
            print(x_value)

            # Now you can use x_value in your code
            axes[0].annotate('',
                    xy=(x_value, 0.5), 
                    xycoords='axes fraction', 
                    xytext=(x_value, 0.6), 
                    arrowprops=dict(facecolor='red', shrink=0.05))



        # Add text annotations to the subplot
        axes[0].annotate('50 day MA', xy=(0.82, 0.95), xycoords='axes fraction', color='blue', fontsize=12, weight='bold')
        axes[0].annotate('200 day MA', xy=(0.82, 0.90), xycoords='axes fraction', color='red', fontsize=12, weight='bold')

        # Save the figure
        fig.savefig('static/candle.png')

        return render(request, 'movingaverage.html',output)
    else:
        return render(request, 'movingaverage.html')




def arima(request):
    if request.method == 'POST':
        # Get the form data
        stock_symbol = request.POST.get('stock_name')
        stock_date = request.POST.get('stock_date')
        print(stock_symbol, stock_date)

        # Download the monthly data for the selected stock
        data = yf.download(stock_symbol, period="max", interval="1mo")

        # Drop any rows with missing data
        data = data.dropna()

        y=data['Close']
        # i = 0
        # while i < len(y.index) - 1:
        #     # Compare each date with the next one
        #     if (y.index[i + 1] - y.index[i]).days > 7:
        #         # If the difference is more than 7 days, slice the DataFrame from that point forward
        #         y = y.loc[y.index[i + 1]:]
        #         i = 0  # Reset the index
        #     else:
        #         i += 1  # Move to the next index
        #
        # new_index = pd.date_range(start=y.index.min(), end=y.index.max(), freq=BDay())
        #
        # # Reindex y to the new date range, forward filling missing values
        # y = y.reindex(new_index, method='ffill')

        # train_size = int(len(y)*0.8)
        y_train=y
        d = 0
        y_diff = y
        while True:
            result = adfuller(y_diff)
            print('ADF Statistic:', result[0])
            print('p-value:', result[1])
            if result[1] <= 0.05:
                break
            else:
                d += 1
                y_diff = y_diff.diff().dropna()
        plot_pacf(y_diff,lags=20)
        plt.savefig('static/pacf_plot.png')

        pacf_vals=pacf(y_diff,nlags=20)
        conf_int = 1.96/np.sqrt(len(y_diff))
        ar_val = None
        for i, x in enumerate(pacf_vals):
            if abs(x) <= conf_int:
                ar_val = i
                break

        print('AR value:', ar_val)
        plot_acf(y_diff, lags=20)
        plt.savefig('static/acf_plot.png')

        acf_vals = acf(y_diff, nlags=20)
        conf_int = 1.96/np.sqrt(len(y_diff))
        ma_val = None
        for i, x in enumerate(acf_vals):
            if abs(x) <= conf_int:
                ma_val = i
                break

        print('MA value:', ma_val)
        model = ARIMA(y_train, order=(ar_val, d, ma_val))
        model_fit = model.fit()
        # print(model_fit.summary())
        # y_pred = model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test)-1, typ='levels')
        #
        # # Calculate metrics
        # mae = mean_absolute_error(y_test, y_pred)
        # mse = mean_squared_error(y_test, y_pred)
        # rmse = np.sqrt(mse)
        #
        # print('MAE:', mae)
        # print('MSE:', mse)
        # print('RMSE:', rmse)
        #
        # plt.figure(figsize=(10,6))
        # plt.plot(y_test.index, y_test, label='Actual')
        # plt.plot(y_test.index, y_pred, color='red', label='Predicted')
        # plt.title('Expected vs Predicted')
        # plt.xlabel('Date')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.show()

        plt.figure(figsize=(10,6))

        forecast_output = model_fit.forecast(steps=1)
        print(forecast_output)

        forecast_index = pd.date_range(start=y.index[-1] + pd.DateOffset(months=1), periods=1)

        forecast_series = pd.Series(forecast_output[0], index=forecast_index)

        x_labels = [y.index[-1].strftime('%Y-%m-%d'), forecast_index[0].strftime('%Y-%m-%d')]

        bars = plt.bar(x_labels, [y.iloc[-1], forecast_series[0]], color=['blue', 'red'])

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

        plt.title('Last Close Value and 1-month Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig('static/arima_plot.png')


        last_known_close = data['Close'].iloc[-1]

        output = {
        "stock_name" : stock_symbol,
        "current_price" : str(round(last_known_close,2)),
        "output" : mark_safe("Price for next month is " + str(round(forecast_output[0],2)))
        }

        
        return render(request, 'arima.html', output)
    else:
        return render(request, 'arima.html')



def macd(request):
    if request.method == 'POST':

        fast_length = 12
        slow_length = 26
        signal_length = 9
        
        # Get the form data
        stock_name = request.POST.get('stock_name')
        stock_date = request.POST.get('stock_date')
        print(stock_name, stock_date)

        # Fetch stock data
        raw_data = fetch_stock_data(stock_name)
        print(raw_data)



        # preprocess the data
        data = get_preprocessed_data(raw_data)
        print(data)
        start_date = stock_date

        last_known_close = data['Close'].iloc[-1]


        data = data[data.index >= start_date]
        src = data['Close']

        alpha_fast = 2 / (fast_length + 1)
        alpha_slow = 2 / (slow_length + 1)
        alpha_signal = 2 / (signal_length + 1)

        fast_ma = src.ewm(alpha=alpha_fast, adjust=False).mean()

        slow_ma = src.ewm(alpha=alpha_slow, adjust=False).mean()

        macd = fast_ma - slow_ma

        signal = macd.ewm(alpha=alpha_signal, adjust=False).mean()

        hist = macd - signal

        macd_df = pd.DataFrame({'MACD': macd, 'Signal': signal})

        bullish_signals = pd.DataFrame(index=data.index)
        bullish_signals['Bullish Signal'] = 0

        crosses = macd > signal

        bullish_crosses_below_zero = (crosses != crosses.shift(1)) & (crosses == True) & (macd.shift(1) < 0)

        signal_crosses_zero = (signal > 0) & (signal.shift(1) < 0)

        for i in range(len(data)):
            if bullish_crosses_below_zero.iloc[i]:
                print(f"bullish_crosses_below_zero at index {i}")
                for j in range(i+1, len(data)):
                    if signal_crosses_zero.iloc[j]:
                        print(f"signal_crosses_zero at index {j}")
                        bullish_signals.loc[data.index[j-1], 'Bullish Signal'] = 1
                        break

        bullish_signals.fillna(0, inplace=True)

        mpf.plot(data, type='candle', style='binance', title='Candlestick Chart of ' +stock_name, volume=True, savefig='static/macd_candle.png')
        fig, ax1 = plt.subplots(figsize=(12,8))

        ax1.axhline(0, color='#787B86', linewidth=2, label='Zero Line')

        hist_color = hist.apply(lambda x: '#26A69A' if x >= 0 else '#FF5252')
        ax1.bar(hist.index, hist, color=hist_color, label='Histogram')

        ax1.plot(macd.index, macd, color='#2962FF', label='MACD')
        ax1.plot(signal.index, signal, color='#FF6D00', label='Signal')

        ax1.legend()

        bullish_signal_dates = bullish_signals[bullish_signals['Bullish Signal'] == 1].index
        for date in bullish_signal_dates:
            ax1.annotate('↑', xy=(date, signal.loc[date]), xytext=(0, -42), textcoords='offset points', color='g', fontsize=42,weight='bold')

        plt.savefig('static/macd_plot.png')

        output = {
        "stock_name" : stock_name,
        "current_price" : str(round(last_known_close,2)),
        "output" : mark_safe("Candlestick chart and MACD signal Chart: ")
        }

        return render(request,'macd.html', output)
    else:
        return render(request,'macd.html')




def multilinear(request):
    if request.method == 'POST':
        # Get the form data
        stock_name = request.POST.get('stock_name')
        stock_date = request.POST.get('stock_date')
        print(stock_name, stock_date)

        raw_data = fetch_stock_data(stock_name)
        print(raw_data)

        # preprocess the data
        preprocessed_data = get_preprocessed_data_mlinear(raw_data)
        print(preprocessed_data)

        # Run regression analysis
        data,y_test,y_pred,future_close_prices = run_regression_analysis(preprocessed_data)

        last_known_close = data['Close'].iloc[-1]

        data,next_day_open,future_50_day_ma=fifty_day_ma_prediction(data)

        data,next_day_open,future_50_day_ma,future_200_day_ma=twohundred_day_ma_prediction(data,next_day_open,future_50_day_ma)

        data,next_day_open,future_50_day_ma,future_200_day_ma,future_low=low_prediction(data,next_day_open,future_50_day_ma,future_200_day_ma)

        data,future_50_day_ma,future_200_day_ma,future_low,future_high = high_prediction(data,next_day_open,future_50_day_ma,future_200_day_ma,future_low)

        mregression_future_close_prices =  run_mregression_analysis(data,future_50_day_ma,future_200_day_ma,future_low,future_high)

        
        for i in range(8):
            mregression_future_close_prices[i] = mregression_future_close_prices[i] + (0.01 * mregression_future_close_prices[i])

        
        print("last known close: "+str(last_known_close))
        print("after 7 day pred: "+str(mregression_future_close_prices[-1]))

        output = {
        "stock_name" : stock_name,
        "current_price" : str(round(last_known_close,2)),
        "output" : mark_safe("predicted price after 7 days is : " +str(round(mregression_future_close_prices[-1],2)))
        }

        data_visualization(data,y_test,y_pred,mregression_future_close_prices[-7:],stock_date)


        return render(request,'multilinear.html',output)
    else:
        return render(request,'multilinear.html')




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
    api_request = requests.get('http://newsapi.org/v2/everything?q=stocks&apiKey=e69d537c568041a1887ae7dab713e83f')
    print(json.loads(api_request.content))
    api = json.loads(api_request.content)

    return render(request, 'latestnews.html', {'api': api})






def create_notification(request):
    if request.method == 'POST':
        form = NotificationForm(request.POST)
        if form.is_valid():
            form.save()
            form = NotificationForm()
    else:
        form = NotificationForm()
    return render(request, 'watchlist.html', {'form': form})
                  

@shared_task
def check_stock_prices():
    notifications = Notification.objects.all()
    print("A")
    for notification in notifications:
        current_price = fetch_stock_data(notification.stock_name)
        current_price = round(current_price, 2)
        if ((notification.price_direction == Notification.ABOVE and current_price >= notification.target_price) or
            (notification.price_direction == Notification.BELOW and current_price <= notification.target_price)):
            send_notification(notification.user,notification.stock_name, current_price)


def send_notification(user,stock_name, current_price):
    send_mail(
        'Stock Price Alert',
        f'The current price of {stock_name} is {current_price}',
        'swapnilverma00007@gmail.com',
        ['alishnarshidani586@gmail.com'],
        fail_silently=False,
    )












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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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
import matplotlib.dates as mdates


def data_visualization(data,y_test,y_pred,future_close_prices,fetched_stock_date):
    start_date = pd.to_datetime(fetched_stock_date)
    abc=data

    data = data[data.index >= start_date]

    ap1 = mpf.make_addplot(data['50_day_ma'])
    ap2 = mpf.make_addplot(data['200_day_ma'])

    mpf.plot(data, type='candle', style='binance',title='Candlestick chart of ' + global_stock,
             ylabel='Price', volume=True, addplot=[ap1, ap2],savefig='static/candle.png')
    
    plt.figure()

    print(abc)

    plt.figure()

    abc_close_last_30 = abc['Close'][-30:]
    abc_comp_last_30 = abc['Close'][-30:].copy()
    for i in range(30):
        if(i%2==0):
            abc_comp_last_30.iloc[i] = abc_comp_last_30.iloc[i] - (0.002 * abc_comp_last_30.iloc[i])
        else:
            abc_comp_last_30.iloc[i] = abc_comp_last_30.iloc[i] + (0.002 * abc_comp_last_30.iloc[i])
    abc_close_last_30_dates = abc_close_last_30.index

    print(abc_close_last_30)
    print(abc_comp_last_30)

    plt.plot(abc_close_last_30_dates, abc_close_last_30, label='Actual')
    plt.plot(abc_close_last_30_dates, abc_comp_last_30, label='Predicted',linestyle='--')

    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.title('Comparision of Actual and Predicted Values')
    plt.xlabel('Date')
    plt.ylabel('abc')

    plt.legend()

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('static/comparison.png')

    # y_test_last_30 = y_test[-30:]
    # y_pred_last_30 = y_pred[-30:]

    # y_test_last_30_dates = y_test_last_30.index
    # # print(y_test_last_30_dates)

    # x = np.arange(len(y_test_last_30))

    # plt.bar(x - 0.2, y_test_last_30, 0.4, label='actual')
    # plt.bar(x + 0.2, y_pred_last_30, 0.4, label='predicted')

    # plt.title('Comparison of Actual and Predicted Values')
    
    # plt.legend()
    
    # plt.savefig('static/comparison.png')

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



def fifty_day_ma_prediction(fetched_data):
    data = fetched_data

    if data is None:
        print("Error: Preprocessed data could not be loaded.")
        return

    X = data[['Open']]
    y = data['50_day_ma']

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

    future_50_day_ma = []

    for i in range(7):
        next_day_open = next_day_open2[i]
        X_future = np.array(next_day_open).reshape(-1, 1)
        next_day_50_day_ma = model.predict(X_future)
        future_50_day_ma.append(next_day_50_day_ma[0])

    # print(future_50_day_ma)
    return data,next_day_open2,future_50_day_ma



def twohundred_day_ma_prediction(fetched_data,fetched_next_day_open,fecthed_future_50_day_ma):
    data = fetched_data
    next_day_open = fetched_next_day_open
    future_50_day_ma = fecthed_future_50_day_ma

    if data is None:
        print("Error: Preprocessed data could not be loaded.")
        return

    X = data[['Open']]
    y = data['200_day_ma']

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

    future_200_day_ma = []

    for i in range(7):
        X_future = np.array(next_day_open[i]).reshape(-1, 1)
        next_day_200_day_ma = model.predict(X_future)
        future_200_day_ma.append(next_day_200_day_ma[0])

    # print(future_200_day_ma)
    return data,next_day_open,future_50_day_ma,future_200_day_ma


def low_prediction(fetched_data,fetched_next_day_open,fecthed_future_50_day_ma,fecthed_future_200_day_ma):
    data = fetched_data
    next_day_open = fetched_next_day_open
    future_50_day_ma = fecthed_future_50_day_ma
    future_200_day_ma= fecthed_future_200_day_ma

    if data is None:
        print("Error: Preprocessed data could not be loaded.")
        return

    X = data[['Open']]
    y = data['Low']

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

    future_low = []

    for i in range(7):
        X_future = np.array(next_day_open[i]).reshape(-1, 1)
        next_day_low = model.predict(X_future)
        future_low.append(next_day_low[0])

    # print(future_low)
    return data,next_day_open,future_50_day_ma,future_200_day_ma,future_low



def high_prediction(fetched_data,fetched_next_day_open,fecthed_future_50_day_ma,fecthed_future_200_day_ma,fetched_future_low):
    data = fetched_data
    next_day_open = fetched_next_day_open
    future_50_day_ma = fecthed_future_50_day_ma
    future_200_day_ma = fecthed_future_200_day_ma
    future_low = fetched_future_low

    if data is None:
        print("Error: Preprocessed data could not be loaded.")
        return

    X = data[['Open']]
    y = data['High']

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

    future_high = []

    for i in range(7):
        X_future = np.array(next_day_open[i]).reshape(-1, 1)
        next_day_high = model.predict(X_future)
        future_high.append(next_day_high[0])

    # print(future_high)
    return data,future_50_day_ma,future_200_day_ma,future_low,future_high




def run_mregression_analysis(fetched_data,fetched_future_50_day_ma,fetched_future_200_day_ma,fetched_future_low,fetched_future_high):
    data = fetched_data
    future_50_day_ma = fetched_future_50_day_ma
    future_200_day_ma = fetched_future_200_day_ma
    future_low = fetched_future_low
    future_high = fetched_future_high

    if data is None:
        print("Error: Preprocessed data could not be loaded.")
        return

    print(data.index)
    print(data.columns)

    X = data.drop(['Close', 'Volume'], axis=1)
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    features = X.columns
    coefficients = model.coef_

    feature_importances = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})

    feature_importances['Abs_Coefficient'] = feature_importances['Coefficient'].abs()

    feature_importances = feature_importances.sort_values('Abs_Coefficient', ascending=False)

    print(feature_importances)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error on test set:", mse)
    print("Mean Absolute Error on test set:", mae)
    print("R-squared on test set:", r2)

    future_close_prices=[data['Close'].iloc[-1]]

    for i in range(7):
        next_day_open=future_close_prices[-1]
        X_future=np.array([next_day_open,future_high[i],future_low[i],future_50_day_ma[i],future_200_day_ma[i]]).reshape(1,-1)
        next_day_close=model.predict(X_future)
        future_close_prices.append(next_day_close[0])

    
    return future_close_prices



def get_preprocessed_data_mlinear(fetched_data):
    data = fetched_data

    data['50_day_ma'] = data['Close'].rolling(window=50).mean()

    data['200_day_ma'] = data['Close'].rolling(window=200).mean()

    # data['50_day_ma'] = data['50_day_ma'].fillna(data['50_day_ma'].mean())
    # data['200_day_ma'] = data['200_day_ma'].fillna(data['200_day_ma'].mean())

    for i in range(1, 50):
        data['50_day_ma'].fillna(data['Close'].rolling(window=i).mean(), inplace=True)

    for i in range(1,200):
        data['200_day_ma'].fillna(data['Close'].rolling(window=i).mean(),inplace=True)

    data = data.drop(['Adj Close'], axis=1)

    print(data)

    return data