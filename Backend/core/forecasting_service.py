import torch
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path
from config.settings import settings
from core.model import HALSTM
from db.postgres_client import get_db

logger = logging.getLogger(__name__)

class OnDemandForecastingService:
    def __init__(self, load_model: bool = True):
        self.device = torch.device('cpu')
        self.model = HALSTM().to(self.device)
        self.target_scaler = None
        self.feature_scaler = None
        self.model_loaded = False
        
        if load_model:
            self._load_model_and_scalers()
            
        self.feature_cols = [
            'item_cnt_month', 'item_price', 'Return',
            'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3',
            'item_cnt_month_mean_shop', 'item_cnt_month_mean_item', 'item_cnt_month_mean_category',
            'shop_id_mean_encode', 'item_id_mean_encode', 'item_category_id_mean_encode'
        ]

    def _load_model_and_scalers(self):
        """Load model and scalers with better error handling and path resolution"""
        try:
            # Try multiple possible paths for the model
            possible_model_paths = [
                settings.get_model_path("best_ha_lstm.pth"),  # From settings
                Path(__file__).parent.parent / "models" / "best_ha_lstm.pth",  # Relative to current file
                Path(__file__).parent / "models" / "best_ha_lstm.pth",  # Same directory
                Path("models/best_ha_lstm.pth"),  # Current working directory
                Path("Backend/models/best_ha_lstm.pth"),  # From project root
            ]
            
            model_path = None
            for path in possible_model_paths:
                if Path(path).exists():
                    model_path = path
                    break
            
            if model_path is None:
                logger.error("Model file not found in any of the expected locations:")
                for path in possible_model_paths:
                    logger.error(f"  - {path}")
                logger.warning("Running in demo mode without pre-trained model")
                return
            
            # Load model
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Loaded model from {model_path}")
            
            # Try multiple possible paths for scalers
            possible_scaler_paths = [
                settings.get_model_path("target_scaler.pkl"),
                Path(model_path).parent / "target_scaler.pkl",
            ]
            
            target_scaler_path = None
            feature_scaler_path = None
            
            for base_path in possible_scaler_paths:
                target_path = Path(str(base_path).replace("target_scaler.pkl", "target_scaler.pkl"))
                feature_path = Path(str(base_path).replace("target_scaler.pkl", "feature_scaler.pkl"))
                
                if target_path.exists() and feature_path.exists():
                    target_scaler_path = target_path
                    feature_scaler_path = feature_path
                    break
            
            if target_scaler_path and feature_scaler_path:
                self.target_scaler = joblib.load(target_scaler_path)
                self.feature_scaler = joblib.load(feature_scaler_path)
                logger.info(f"Loaded scalers from {target_scaler_path.parent}")
                self.model_loaded = True
            else:
                logger.error("Scaler files not found")
                logger.warning("Running in demo mode without scalers")
                
        except Exception as e:
            logger.error(f"Failed to load model or scalers: {str(e)}")
            logger.warning("Running in demo mode - predictions will be simulated")

    def _generate_demo_prediction(self, shop_id: int, item_id: int, months: int = 3):
        """Generate demo predictions when model is not loaded"""
        logger.info(f"Generating demo predictions for shop_id={shop_id}, item_id={item_id}")
        
        predictions = []
        insights = []
        base_date = datetime.now()
        
        # Simulate predictions based on shop_id and item_id for consistency
        np.random.seed(shop_id * 1000 + item_id)  # Consistent seed
        base_demand = max(10, int((shop_id + item_id) % 100))
        
        for step in range(months):
            # Add some random variation
            variation = np.random.normal(0, 0.1)
            predicted_demand = max(1, int(base_demand * (1 + variation)))
            unit_price = 50 + (item_id % 50)  # Simulate price based on item_id
            
            predictions.append({
                'shop_id': shop_id,
                'shop_name': f"Demo_Shop_{shop_id}",
                'item_id': item_id,
                'item_category_name': f"Category_{item_id % 10}",
                'predicted_product_demand': predicted_demand,
                'predicted_total_sales': predicted_demand * unit_price
            })
            
            insights.append({
                'timestamp_reference': (base_date + timedelta(days=30 * step)).isoformat(),
                'shop_id': shop_id,
                'shop_name': f"Demo_Shop_{shop_id}",
                'item_id': item_id,
                'forecasted_value_unscaled': float(predicted_demand),
                'lstm_trend_reliance': 0.7,  # Demo value
                'attention_t_minus_2': 0.2,  # Demo value
                'attention_t_minus_1': 0.3,  # Demo value
                'attention_t_minus_0': 0.5   # Demo value
            })
        
        return {
            'predictions': predictions,
            'insights': insights,
            'demo_mode': True
        }

    async def _fetch_historical_data(self, shop_id: int, item_id: int, db):
        """Fetch historical data with fallback to demo data"""
        try:
            query = """
                SELECT date, shop_id, shop_name, item_id, item_category_name, 
                       monthly_revenue, item_cnt_month
                FROM historical_monthly_sales
                WHERE shop_id = $1 AND item_id = $2
                ORDER BY date DESC
                LIMIT 3
            """
            results = await db.fetch(query, shop_id, item_id)
            df = pd.DataFrame([dict(row) for row in results])
            
            if len(df) < 3:
                logger.warning(f"Insufficient data for shop_id={shop_id}, item_id={item_id}. Padding with zeros.")
                shop_name = df['shop_name'].iloc[0] if not df.empty else f"Shop_{shop_id}"
                item_category_name = df['item_category_name'].iloc[0] if not df.empty else "Unknown"
                while len(df) < 3:
                    df = pd.concat([pd.DataFrame([{
                        'date': df['date'].min() - timedelta(days=30) if not df.empty else datetime.now(),
                        'shop_id': shop_id,
                        'shop_name': shop_name,
                        'item_id': item_id,
                        'item_category_name': item_category_name,
                        'monthly_revenue': 0.0,
                        'item_cnt_month': 0
                    }]), df], ignore_index=True)
            
            df = df.sort_values('date')
            df['item_price'] = df['monthly_revenue'] / df['item_cnt_month'].replace(0, 1)
            df['Return'] = 0.0
            
            for lag in range(1, 4):
                df[f'item_cnt_month_lag_{lag}'] = df['item_cnt_month'].shift(lag).fillna(0)
            
            agg_query = """
                SELECT 
                    AVG(item_cnt_month) FILTER (WHERE shop_id = $1) AS shop_mean,
                    AVG(item_cnt_month) FILTER (WHERE item_id = $2) AS item_mean,
                    AVG(item_cnt_month) FILTER (WHERE item_category_name = $3) AS category_mean
                FROM historical_monthly_sales
            """
            agg_results = await db.fetch(agg_query, shop_id, item_id, df['item_category_name'].iloc[0])
            aggregates = dict(agg_results[0])
            
            df['item_cnt_month_mean_shop'] = aggregates['shop_mean'] or 0.0
            df['item_cnt_month_mean_item'] = aggregates['item_mean'] or 0.0
            df['item_cnt_month_mean_category'] = aggregates['category_mean'] or 0.0
            df['shop_id_mean_encode'] = aggregates['shop_mean'] or 0.0
            df['item_id_mean_encode'] = aggregates['item_mean'] or 0.0
            df['item_category_id_mean_encode'] = aggregates['category_mean'] or 0.0
            
            if self.feature_scaler is not None:
                numerical = df[self.feature_cols].to_numpy().astype(np.float32)
                numerical = np.nan_to_num(numerical, nan=0.0)
                numerical = self.feature_scaler.transform(numerical)
                numerical = numerical.reshape(1, 3, len(self.feature_cols))
            else:
                numerical = None
            
            return numerical, df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return None, pd.DataFrame()

    async def generate_forecast_with_insights(self, shop_id: int, item_id: int, months: int = 3):
        """Generate forecasts with fallback to demo mode"""
        if not self.model_loaded:
            logger.warning("Model not loaded, using demo mode")
            return self._generate_demo_prediction(shop_id, item_id, months)
        
        try:
            async with get_db() as db:
                numerical, df = await self._fetch_historical_data(shop_id, item_id, db)
                
                if numerical is None or self.target_scaler is None:
                    logger.warning("Falling back to demo mode due to data/scaler issues")
                    return self._generate_demo_prediction(shop_id, item_id, months)
                
                input_tensor = torch.tensor(numerical, dtype=torch.float32).to(self.device)
                
                predictions = []
                insights = []
                current_sequence = numerical.copy()
                base_date = df['date'].max() if not df.empty else datetime.now()
                shop_name = df['shop_name'].iloc[0] if not df.empty else f"Shop_{shop_id}"
                item_category_name = df['item_category_name'].iloc[0] if not df.empty else "Unknown"
                unit_price = df['item_price'].mean() if not df.empty else 100.0
                
                with torch.no_grad():
                    for step in range(months):
                        input_tensor = torch.tensor(current_sequence, dtype=torch.float32).to(self.device)
                        output, attn_dict = self.model(input_tensor)
                        
                        unscaled_pred = self.target_scaler.inverse_transform(output.cpu().numpy())
                        predicted_demand = max(0, round(unscaled_pred[0][0]))
                        predictions.append({
                            'shop_id': shop_id,
                            'shop_name': shop_name,
                            'item_id': item_id,
                            'item_category_name': item_category_name,
                            'predicted_product_demand': predicted_demand,
                            'predicted_total_sales': predicted_demand * unit_price
                        })
                        
                        mha_weights = attn_dict['mha_weights'][:, -1, :].cpu().numpy()[0]
                        gate_weights = attn_dict['gate_weights'].cpu().numpy()[0]
                        insights.append({
                            'timestamp_reference': (base_date + timedelta(days=30 * step)).isoformat(),
                            'shop_id': shop_id,
                            'shop_name': shop_name,
                            'item_id': item_id,
                            'forecasted_value_unscaled': float(predicted_demand),
                            'lstm_trend_reliance': float(np.mean(gate_weights)),
                            'attention_t_minus_2': float(mha_weights[0]),
                            'attention_t_minus_1': float(mha_weights[1]),
                            'attention_t_minus_0': float(mha_weights[2])
                        })
                        
                        new_features = current_sequence[0, -1, :].copy()
                        new_features[0] = self.target_scaler.transform([[predicted_demand]])[0, 0]
                        current_sequence = np.roll(current_sequence, -1, axis=1)
                        current_sequence[0, -1, :] = new_features
                
                return {
                    'predictions': predictions,
                    'insights': insights
                }
        
        except Exception as e:
            logger.error(f"Error in generate_forecast_with_insights: {str(e)}")
            return self._generate_demo_prediction(shop_id, item_id, months)

    async def batch_generate_forecasts(self, requests: List[dict]):
        """Batch generate forecasts with fallback to demo mode"""
        if not self.model_loaded:
            logger.warning("Model not loaded, using demo mode for batch predictions")
            return [
                {
                    'shop_id': req['shop_id'],
                    'item_id': req['item_id'],
                    'months': req['months'],
                    'predictions': self._generate_demo_prediction(
                        req['shop_id'], req['item_id'], req['months']
                    )['predictions'],
                    'confidence': 0.5,  # Lower confidence for demo
                    'demo_mode': True
                }
                for req in requests
            ]
        
        try:
            results = []
            async with get_db() as db:
                shop_item_pairs = [(req['shop_id'], req['item_id'], req['months']) for req in requests]
                data_cache = {}
                
                for shop_id, item_id, months in shop_item_pairs:
                    try:
                        cache_key = f"{shop_id}:{item_id}"
                        if cache_key not in data_cache:
                            numerical, df = await self._fetch_historical_data(shop_id, item_id, db)
                            data_cache[cache_key] = (numerical, df)
                        
                        numerical, df = data_cache[cache_key]
                        
                        if numerical is None:
                            # Fallback to demo for this specific item
                            demo_result = self._generate_demo_prediction(shop_id, item_id, months)
                            results.append({
                                'shop_id': shop_id,
                                'item_id': item_id,
                                'months': months,
                                'predictions': demo_result['predictions'],
                                'confidence': 0.5,
                                'demo_mode': True
                            })
                            continue
                        
                        input_tensor = torch.tensor(numerical, dtype=torch.float32).to(self.device)
                        
                        predictions = []
                        current_sequence = numerical.copy()
                        base_date = df['date'].max() if not df.empty else datetime.now()
                        shop_name = df['shop_name'].iloc[0] if not df.empty else f"Shop_{shop_id}"
                        item_category_name = df['item_category_name'].iloc[0] if not df.empty else "Unknown"
                        unit_price = df['item_price'].mean() if not df.empty else 100.0
                        
                        with torch.no_grad():
                            for step in range(months):
                                input_tensor = torch.tensor(current_sequence, dtype=torch.float32).to(self.device)
                                output, attn_dict = self.model(input_tensor)
                                
                                unscaled_pred = self.target_scaler.inverse_transform(output.cpu().numpy())
                                predicted_demand = max(0, round(unscaled_pred[0][0]))
                                predictions.append({
                                    'shop_id': shop_id,
                                    'shop_name': shop_name,
                                    'item_id': item_id,
                                    'item_category_name': item_category_name,
                                    'predicted_product_demand': predicted_demand,
                                    'predicted_total_sales': predicted_demand * unit_price
                                })
                                
                                new_features = current_sequence[0, -1, :].copy()
                                new_features[0] = self.target_scaler.transform([[predicted_demand]])[0, 0]
                                current_sequence = np.roll(current_sequence, -1, axis=1)
                                current_sequence[0, -1, :] = new_features
                        
                        results.append({
                            'shop_id': shop_id,
                            'item_id': item_id,
                            'months': months,
                            'predictions': predictions,
                            'confidence': 0.85
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing shop_id={shop_id}, item_id={item_id}: {str(e)}")
                        # Fallback to demo for this specific item
                        demo_result = self._generate_demo_prediction(shop_id, item_id, months)
                        results.append({
                            'shop_id': shop_id,
                            'item_id': item_id,
                            'months': months,
                            'predictions': demo_result['predictions'],
                            'confidence': 0.3,
                            'demo_mode': True,
                            'error': str(e)
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Critical error in batch_generate_forecasts: {str(e)}")
            # Return demo results for all requests
            return [
                {
                    'shop_id': req['shop_id'],
                    'item_id': req['item_id'],
                    'months': req['months'],
                    'predictions': self._generate_demo_prediction(
                        req['shop_id'], req['item_id'], req['months']
                    )['predictions'],
                    'confidence': 0.3,
                    'demo_mode': True,
                    'error': str(e)
                }
                for req in requests
            ]

# Initialize with graceful fallback
forecasting_service = OnDemandForecastingService(load_model=True)