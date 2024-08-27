from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
from datetime import datetime, timedelta
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import requests
import sqlite3

app = Flask(__name__)

def load_data_and_model(n_days):
    if n_days == 1:
        model = load_model('models/lstm_btc_V2.h5')
        end_date = datetime.now(pytz.timezone('Asia/Kolkata'))
        start_date = end_date - timedelta(hours=60)
        interval = '1h'
    else:
        model = load_model('models/lstm_btc_V1.h5')
        end_date = datetime.now(pytz.timezone('Asia/Kolkata'))
        start_date = end_date - timedelta(days=60)
        interval = '1d'

    data = yf.download('BTC-USD', start=start_date, end=end_date, interval=interval)
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize(pytz.utc).tz_convert('Asia/Kolkata')
    else:
        data.index = data.index.tz_convert('Asia/Kolkata')

    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    sequence_length = min(len(scaled_data) - 1, 60)
    X = np.array([scaled_data[i:i + sequence_length] for i in range(len(scaled_data) - sequence_length)])
    if X.ndim == 2:
        X = X.reshape((X.shape[0], X.shape[1], 1))

    last_data = scaled_data[-sequence_length:].reshape((1, sequence_length, 1))

    return model, last_data, scaler, data.index[-1], data['Close'][-1]

def forecast_future_prices(model, last_data, n_periods, scaler, last_date, period='hours'):
    predictions = []
    dates = []
    current_date = last_date

    for _ in range(n_periods):
        pred_scaled = model.predict(last_data)
        pred = scaler.inverse_transform(pred_scaled).flatten()
        predictions.append(pred[0])

        if period == 'hours':
            current_date = current_date + pd.Timedelta(hours=1)
        else:
            current_date = current_date + pd.Timedelta(days=1)

        dates.append(current_date)
        last_data = np.roll(last_data, shift=-1, axis=1)
        last_data[0, -1, 0] = pred_scaled[0, 0]

    return dates, predictions

def load_initial_data():
    end_date = datetime.now(pytz.timezone('Asia/Kolkata'))
    start_date = end_date - timedelta(days=60)
    data = yf.download('BTC-USD', start=start_date, end=end_date, interval='1d')
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize(pytz.utc).tz_convert('Asia/Kolkata')
    else:
        data.index = data.index.tz_convert('Asia/Kolkata')

    return data[['Close']]

def fetch_bitcoin_news():
    api_key = 'e3b1d824a61b4815ae3c74031b026ae1'
    url = f'https://newsapi.org/v2/everything?q=bitcoin&apiKey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data.get('articles', [])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    initial_data = load_initial_data()
    dates = initial_data.index
    prices = initial_data['Close']
    
    latest_price = prices[-1]
    previous_price = prices[-2] if len(prices) > 1 else latest_price
    
    price_trend = "▲" if latest_price > previous_price else "▼"
    color = "green" if latest_price > previous_price else "red"
    
    # Calculate percentage change
    percentage_change = ((latest_price - previous_price) / previous_price) * 100 if previous_price > 0 else 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices,
                             mode='lines+markers', marker=dict(color='blue'), line=dict(width=2)))

    fig.update_layout(
        title='Bitcoin Prices for Last 60 Days',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_tickformat='%Y-%m-%d',
        xaxis=dict(range=[dates[0], dates[-1]])
    )

    fig.update_xaxes(
        tickangle=-45,
        rangeslider_visible=True
    )

    graph = fig.to_html(full_html=False)

    return render_template('index.html', graph=graph, latest_price=f"${latest_price:,.2f}", price_trend=price_trend, color=color, percentage_change=f"{percentage_change:.2f}")

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    graph = None
    prediction = None
    arrow = None
    color = None
    percentage_change = None

    if request.method == 'POST':
        try:
            n_days = int(request.form.get('n_days'))
            model, last_data, scaler, last_date, latest_price = load_data_and_model(n_days)
            n_periods = n_days if n_days > 1 else n_days * 24
            period = 'days' if n_days > 1 else 'hours'

            dates, predictions = forecast_future_prices(model, last_data, n_periods, scaler, last_date, period)

            # Calculate percentage change
            if latest_price is not None and len(predictions) > 0:
                percentage_change = ((predictions[-1] - latest_price) / latest_price) * 100
            else:
                percentage_change = 0  # Default to 0 if None

            arrow = "▲" if percentage_change > 0 else "▼" if percentage_change < 0 else None
            color = "green" if percentage_change > 0 else "red" if percentage_change < 0 else "gray"

            forecast_df = pd.DataFrame({'Datetime': dates, 'Predicted Price': predictions})

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=forecast_df['Datetime'], y=forecast_df['Predicted Price'],
                                     mode='lines+markers', marker=dict(color='blue'), line=dict(width=2)))

            if n_days == 1:
                fig.update_layout(
                    title='Bitcoin Price Forecast for Next 24 Hours',
                    xaxis_title='Time of Day',
                    yaxis_title='Predicted Price',
                    xaxis_tickformat='%H:%M',
                    xaxis=dict(range=[dates[0], dates[-1]])
                )
            else:
                fig.update_layout(
                    title=f'Bitcoin Price Forecast for Next {n_days} Days',
                    xaxis_title='Date',
                    yaxis_title='Predicted Price',
                    xaxis=dict(range=[dates[0], dates[-1]])
                )

                fig.update_xaxes(
                    tickformat='%Y-%m-%d',
                    dtick='D1' if n_days <= 5 else ('D2' if n_days <= 15 else ('D5' if n_days <= 30 else 'M1')),
                    tickangle=-45,
                    tickvals=forecast_df['Datetime'][::max(1, len(forecast_df) // 20)],
                    ticktext=[date.strftime('%Y-%m-%d') for date in forecast_df['Datetime'][::max(1, len(forecast_df) // 20)]]
                )

            fig.update_xaxes(rangeslider_visible=True)

            graph = fig.to_html(full_html=False)

            # Ensure prediction and percentage_change are formatted safely
            prediction = f"${predictions[-1]:,.2f} ({percentage_change:.2f}%)" if percentage_change is not None else None

        except Exception as e:
            print(f"Error: {e}")

    return render_template('forecast.html', graph=graph, prediction=prediction, arrow=arrow, color=color, percentage_change=f"{percentage_change:.2f}" if percentage_change is not None else "N/A")

@app.route('/news')
def news():
    articles = fetch_bitcoin_news()

    def format_datetime(iso_str):
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return dt.strftime('%B %d, %Y at %I:%M %p')  # Example: July 29, 2024 at 04:21 PM

    return render_template('news.html', articles=articles, format_datetime=format_datetime)

@app.route("/contact",methods=['GET','POST'])
def contactus():
    if request.method=="POST":
        fname = request.form.get("full_name")
        pno   = request.form.get("phone_number")
        email = request.form.get("email")
        addr   = request.form.get("address")
        msg  = request.form.get("message")
        conn = sqlite3.connect('contact.db')
        cur = conn.cursor()

        # Correct SQL statement with placeholders
        cur.execute('''
            INSERT INTO contacts (FullName, PhoneNumber, Email, Address, Message)
            VALUES (?, ?, ?, ?, ?)
        ''', (fname, pno, email, addr, msg))

        conn.commit()

        return render_template('message.html')
    
        
         

    else:
        return render_template('contactus.html')
    

if __name__ == '__main__':
    app.run(debug=True)
