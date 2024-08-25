import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
from datetime import datetime, timedelta
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import requests

# Function to load data and model based on user selection
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

# Function to forecast future prices
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

# Load past 60 days of Bitcoin prices on app load
def load_initial_data():
    end_date = datetime.now(pytz.timezone('Asia/Kolkata'))
    start_date = end_date - timedelta(days=60)
    data = yf.download('BTC-USD', start=start_date, end=end_date, interval='1d')
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize(pytz.utc).tz_convert('Asia/Kolkata')
    else:
        data.index = data.index.tz_convert('Asia/Kolkata')

    return data[['Close']]

# Fetch Bitcoin news from an API
def fetch_bitcoin_news():
    api_key = 'e3b1d824a61b4815ae3c74031b026ae1'
    url = f'https://newsapi.org/v2/everything?q=bitcoin&apiKey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data.get('articles', [])

# Streamlit UI
st.set_page_config(page_title='Bitcoin Price Forecasting', page_icon=':moneybag:', layout='wide')

# Create a sidebar menu
menu = st.sidebar.radio("Select a Page", ["Home", "News", "Contact Me"])

if menu == "Home":
    st.title('Bitcoin Price Forecasting')

    # Container for percentage change and buttons
    with st.container():
        st.markdown("<h2 style='text-align: center;'>Select Forecast Interval:</h2>", unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button('1 Day'):
                n_days = 1
                forecast_mode = True
        with col2:
            if st.button('5 Days'):
                n_days = 5
                forecast_mode = True
        with col3:
            if st.button('15 Days'):
                n_days = 15
                forecast_mode = True
        with col4:
            if st.button('20 Days'):
                n_days = 20
                forecast_mode = True
        with col5:
            if st.button('30 Days'):
                n_days = 30
                forecast_mode = True

        if 'forecast_mode' in locals():
            model, last_data, scaler, last_date, latest_price = load_data_and_model(n_days)
            n_periods = n_days if n_days > 1 else n_days * 24
            period = 'days' if n_days > 1 else 'hours'
            
            dates, predictions = forecast_future_prices(model, last_data, n_periods, scaler, last_date, period)
            
            # Calculate percentage increase/decrease
            percentage_change = ((predictions[-1] - latest_price) / latest_price) * 100
            arrow = "▲" if percentage_change > 0 else "▼"
            color = "green" if percentage_change > 0 else "red"

            # Display the percentage change above the chart
            st.markdown(f"<h3 style='color:{color}; text-align: center;'>{arrow} ${predictions[-1]:,.2f} ({percentage_change:.2f}%)</h3>", unsafe_allow_html=True)

            # Update the chart with the forecast data
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
                    xaxis=dict(range=[dates[0], dates[-1]])  # Ensure x-axis range is correctly set
                )
            else:
                fig.update_layout(
                    title=f'Bitcoin Price Forecast for Next {n_days} Days',
                    xaxis_title='Date',
                    yaxis_title='Predicted Price',
                    xaxis=dict(range=[dates[0], dates[-1]])  # Ensure x-axis range is correctly set
                )
                
                fig.update_xaxes(
                    tickformat='%Y-%m-%d',
                    dtick='D1' if n_days <= 5 else ('D2' if n_days <= 15 else ('D5' if n_days <= 30 else 'M1')),
                    tickangle=-45,
                    tickvals=forecast_df['Datetime'][::max(1, len(forecast_df) // 20)],
                    ticktext=[date.strftime('%Y-%m-%d') for date in forecast_df['Datetime'][::max(1, len(forecast_df) // 20)]]
                )

            fig.update_xaxes(rangeslider_visible=True)
            
            # Replace the initial chart with the forecast chart
            st.plotly_chart(fig, use_container_width=True, height=600)
        else:
            # Display the initial 60 days of data
            initial_data = load_initial_data()
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=initial_data.index, y=initial_data['Close'],
                                     mode='lines+markers', marker=dict(color='blue'), line=dict(width=2)))

            fig.update_layout(
                title='Bitcoin Prices for Last 60 Days',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_tickformat='%Y-%m-%d',
                xaxis=dict(range=[initial_data.index[0], initial_data.index[-1]])
            )

            fig.update_xaxes(
                tickangle=-45,
                rangeslider_visible=True
            )

            # Display the initial chart
            st.plotly_chart(fig, use_container_width=True, height=600)

elif menu == "News":
    st.title('Bitcoin News')

    articles = fetch_bitcoin_news()
    from datetime import datetime

    def format_datetime(iso_str):
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return dt.strftime('%B %d, %Y at %I:%M %p')  # Example: July 29, 2024 at 04:21 PM

# In your Streamlit app

    if articles:
        for article in articles:
            st.subheader(article['title'])
            st.write(article['description'])
            st.write(f"[Read more]({article['url']})")
            st.write(f"Published at: {format_datetime(article['publishedAt'])}")

            st.markdown("---")
    else:
        st.write("No news articles found.")

elif menu == "Contact Me":
    st.title('Contact Me')
    st.write("Feel free to reach out for any queries or collaboration.")
    st.write("Email: bathalapalli9920@gmail.com")
