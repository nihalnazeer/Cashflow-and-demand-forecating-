import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime, timedelta
from config.settings import settings
from core.model import HALSTM
from db.postgres_client import get_db
import asyncio
import aiofiles

logger = logging.getLogger(__name__)

async def fetch_training_data():
    async with get_db() as db:
        query = """
            SELECT date, shop_id, shop_name, item_id, item_category_name, 
                   monthly_revenue, item_cnt_month
            FROM historical_monthly_sales
            ORDER BY date
        """
        results = await db.fetch(query)
        df = pd.DataFrame([dict(row) for row in results])
        
        # Feature engineering (matches forecasting_service.py)
        df['item_price'] = df['monthly_revenue'] / df['item_cnt_month'].replace(0, 1)
        df['Return'] = 0.0
        for lag in range(1, 4):
            df[f'item_cnt_month_lag_{lag}'] = df.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(lag).fillna(0)
        
        df['item_cnt_month_mean_shop'] = df.groupby('shop_id')['item_cnt_month'].transform('mean').fillna(0)
        df['item_cnt_month_mean_item'] = df.groupby('item_id')['item_cnt_month'].transform('mean').fillna(0)
        df['item_cnt_month_mean_category'] = df.groupby('item_category_name')['item_cnt_month'].transform('mean').fillna(0)
        df['shop_id_mean_encode'] = df['item_cnt_month_mean_shop']
        df['item_id_mean_encode'] = df['item_cnt_month_mean_item']
        df['item_category_id_mean_encode'] = df['item_cnt_month_mean_category']
        
        return df

def train_model(df):
    feature_cols = [
        'item_cnt_month', 'item_price', 'Return',
        'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3',
        'item_cnt_month_mean_shop', 'item_cnt_month_mean_item', 'item_cnt_month_mean_category',
        'shop_id_mean_encode', 'item_id_mean_encode', 'item_category_id_mean_encode'
    ]
    
    device = torch.device('cpu')
    model = HALSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Prepare sequences
    sequences = []
    for _, group in df.groupby(['shop_id', 'item_id']):
        group = group.sort_values('date')
        if len(group) >= 3:
            numerical = group[feature_cols].to_numpy()[-3:].astype(np.float32)
            target = group['item_cnt_month'].to_numpy()[-3:].astype(np.float32)
            sequences.append((numerical, target))
    
    if not sequences:
        raise ValueError("Insufficient data for training")
    
    X = np.array([seq[0] for seq in sequences])
    y = np.array([seq[1] for seq in sequences])
    X_reshaped = X.reshape(-1, X.shape[-1])
    y_reshaped = y.reshape(-1, 1)
    
    feature_scaler.fit(X_reshaped)
    target_scaler.fit(y_reshaped)
    X_scaled = feature_scaler.transform(X_reshaped).reshape(X.shape)
    y_scaled = target_scaler.transform(y_reshaped).reshape(y.shape)
    
    # Training loop
    model.train()
    for epoch in range(10):  # Adjust epochs as needed
        total_loss = 0
        for X_batch, y_batch in zip(X_scaled, y_scaled):
            X_tensor = torch.tensor(X_batch, dtype=torch.float32).unsqueeze(0).to(device)
            y_tensor = torch.tensor(y_batch, dtype=torch.float32).unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            output, _ = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        logger.info(f"Epoch {epoch+1}, Loss: {total_loss / len(sequences):.4f}")
    
    # Save model and scalers
    model_dir = settings.MODELS_DIR
    model_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_dir / "best_ha_lstm.pth")
    joblib.dump(target_scaler, model_dir / "target_scaler.pkl")
    joblib.dump(feature_scaler, model_dir / "feature_scaler.pkl")
    logger.info(f"Saved model and scalers to {model_dir}")

async def run_forecast():
    # Check if model files exist
    model_files = ["best_ha_lstm.pth", "target_scaler.pkl", "feature_scaler.pkl"]
    missing_files = [f for f in model_files if not (settings.MODELS_DIR / f).exists()]
    
    if missing_files:
        logger.info(f"Missing model files: {missing_files}. Training new model...")
        df = await fetch_training_data()
        train_model(df)
    
    # Example: Generate forecasts for all shop-item pairs
    async with get_db() as db:
        query = "SELECT DISTINCT shop_id, item_id FROM historical_monthly_sales"
        pairs = await db.fetch(query)
        requests = [{"shop_id": row['shop_id'], "item_id": row['item_id'], "months": 3} for row in pairs]
    
    from core.forecasting_service import forecasting_service
    results = await forecasting_service.batch_generate_forecasts(requests)
    
    # Save results to CSV
    output_dir = settings.RESULTS_DIR
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"forecast_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    forecast_data = []
    for result in results:
        for pred in result['predictions']:
            forecast_data.append({
                'shop_id': pred['shop_id'],
                'shop_name': pred['shop_name'],
                'item_id': pred['item_id'],
                'item_category_name': pred['item_category_name'],
                'predicted_product_demand': pred['predicted_product_demand'],
                'predicted_total_sales': pred['predicted_total_sales']
            })
    
    df = pd.DataFrame(forecast_data)
    async with aiofiles.open(output_file, mode='w') as f:
        await f.write(df.to_csv(index=False))
    logger.info(f"Saved forecast results to {output_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_forecast())