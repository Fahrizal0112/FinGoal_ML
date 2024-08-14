from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime

app = Flask(__name__)

model = load_model('forecast.h5')

scaler = MinMaxScaler(feature_range=(0, 1))

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.json
    ticker = data.get('ticker')
    num_future_steps = data.get('num_future_steps', 4)  
    
    if not ticker:
        return jsonify({'error': 'Ticker symbol is required'}), 400

    end_date = datetime.today().strftime('%Y-%m-%d')

    try:
        stock_data = yf.download(ticker, start='2010-01-01', end=end_date)
    except Exception as e:
        return jsonify({'error': f'Error fetching data: {str(e)}'}), 500
    
    if stock_data.empty:
        return jsonify({'error': 'No data found for the ticker'}), 404
    
    data_close = stock_data[['Close']]
    
    scaled_data = scaler.fit_transform(data_close)
    
    time_step = 60
    last_data = scaled_data[-time_step:]
    current_step = last_data.reshape(1, time_step, 1)

    future_predictions = []
    for _ in range(num_future_steps):
        predicted_price = model.predict(current_step)
        future_predictions.append(predicted_price[0, 0])
        current_step = np.concatenate((current_step[:, 1:, :], predicted_price.reshape(1, 1, 1)), axis=1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    last_date = pd.to_datetime(stock_data.index[-1])
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, num_future_steps + 1)]

    # Calculate monthly returns
    monthly_returns = [
        ((future_predictions[i].item() - future_predictions[i - 1].item()) / future_predictions[i - 1].item()) * 100
        for i in range(1, len(future_predictions))
    ]
    monthly_returns = [0.0] + monthly_returns

    future_predictions_list = future_predictions.flatten().tolist()
    
    monthly_returns_list = [f"{return_value:.2f}%" for return_value in monthly_returns]

    response_data = {
        'ticker': ticker,
        'future_dates': [date.strftime('%Y-%m') for date in future_dates],
        'predicted_prices': future_predictions_list,
        'monthly_returns': monthly_returns_list
    }

    return jsonify(response_data)




if __name__ == '__main__':
    app.run(debug=True)
