from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import io
import json
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files supported'}), 400
        
        df = pd.read_csv(file)
        
        if len(df) < 10:
            return jsonify({'error': 'Need at least 10 rows'}), 400
        
        # Find price column
        price_column = None
        for col in df.columns:
            if 'close' in col.lower():
                price_column = col
                break
        if not price_column:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return jsonify({'error': 'No numeric columns found'}), 400
            price_column = numeric_cols[-1]
        
        prices = pd.to_numeric(df[price_column], errors='coerce').dropna().values
        
        if len(prices) < 10:
            return jsonify({'error': 'Not enough valid data'}), 400
        
        # Statistics
        mean_price = float(np.mean(prices))
        stdev_price = float(np.std(prices))
        min_price = float(np.min(prices))
        max_price = float(np.max(prices))
        
        # Feature engineering
        features = []
        for i in range(5, len(prices) - 1):
            ma5 = float(np.mean(prices[i-5:i]))
            ma10 = float(np.mean(prices[max(0, i-10):i]))
            volatility = float(np.std(prices[max(0, i-5):i]))
            
            features.append({
                'ma5': ma5,
                'ma10': ma10,
                'volatility': volatility,
                'price': float(prices[i+1])
            })
        
        if len(features) < 20:
            return jsonify({'error': 'Not enough data for training'}), 400
        
        # Prepare data
        X = np.array([[f['ma5'], f['ma10'], f['volatility']] for f in features])
        y = np.array([f['price'] for f in features])
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Model 1: Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        
        rmse_lr = float(np.sqrt(mean_squared_error(y_test, y_pred_lr)))
        mae_lr = float(mean_absolute_error(y_test, y_pred_lr))
        r2_lr = float(r2_score(y_test, y_pred_lr))
        
        # Model 2: Random Forest
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        rmse_rf = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
        mae_rf = float(mean_absolute_error(y_test, y_pred_rf))
        r2_rf = float(r2_score(y_test, y_pred_rf))
        
        # Model 3: Ensemble
        y_pred_ensemble = (y_pred_lr + y_pred_rf) / 2
        
        rmse_ensemble = float(np.sqrt(mean_squared_error(y_test, y_pred_ensemble)))
        mae_ensemble = float(mean_absolute_error(y_test, y_pred_ensemble))
        r2_ensemble = float(r2_score(y_test, y_pred_ensemble))
        
        results = {
            'statistics': {
                'count': int(len(prices)),
                'mean': mean_price,
                'stdev': stdev_price,
                'min': min_price,
                'max': max_price
            },
            'models': {
                'Linear Regression': {
                    'rmse': rmse_lr,
                    'mae': mae_lr,
                    'r2': r2_lr,
                    'predictions': y_pred_lr.tolist()
                },
                'Random Forest': {
                    'rmse': rmse_rf,
                    'mae': mae_rf,
                    'r2': r2_rf,
                    'predictions': y_pred_rf.tolist()
                },
                'Ensemble': {
                    'rmse': rmse_ensemble,
                    'mae': mae_ensemble,
                    'r2': r2_ensemble,
                    'predictions': y_pred_ensemble.tolist()
                }
            },
            'actual_prices': y_test.tolist(),
            'all_prices': prices.tolist()
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-csv', methods=['POST'])
def download_csv():
    data = request.json
    df = pd.DataFrame({
        'Index': range(1, len(data['actual']) + 1),
        'Actual': data['actual'],
        'Linear_Regression': data['linear_reg'],
        'Random_Forest': data['random_forest'],
        'Ensemble': data['ensemble']
    })
    
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='predictions.csv'
    )

@app.route('/api/download-report', methods=['POST'])
def download_report():
    data = request.json
    report = f"""STOCK PRICE PREDICTION - ML PROJECT REPORT
============================================

DATASET STATISTICS:
- Total Data Points: {data['stats']['count']}
- Mean Price: ${data['stats']['mean']:.2f}
- Std Deviation: ${data['stats']['stdev']:.2f}
- Min Price: ${data['stats']['min']:.2f}
- Max Price: ${data['stats']['max']:.2f}

MODEL PERFORMANCE:

1. LINEAR REGRESSION
   - RMSE: {data['models']['Linear Regression']['rmse']:.6f}
   - MAE: {data['models']['Linear Regression']['mae']:.6f}
   - R² Score: {data['models']['Linear Regression']['r2']:.6f}

2. RANDOM FOREST
   - RMSE: {data['models']['Random Forest']['rmse']:.6f}
   - MAE: {data['models']['Random Forest']['mae']:.6f}
   - R² Score: {data['models']['Random Forest']['r2']:.6f}

3. ENSEMBLE MODEL
   - RMSE: {data['models']['Ensemble']['rmse']:.6f}
   - MAE: {data['models']['Ensemble']['mae']:.6f}
   - R² Score: {data['models']['Ensemble']['r2']:.6f}

BEST MODEL: {data['best_model']}

Generated by Stock Price Prediction ML Tool
"""
    
    return send_file(
        io.BytesIO(report.encode()),
        mimetype='text/plain',
        as_attachment=True,
        download_name='report.txt'
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)