import time
import torch
import torch.nn as nn
import numpy as np
import polars as pl
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class SalesDataset(Dataset):
    def __init__(self, X_file, y_file, sequence_length=12, num_shops=60, num_items=22170, num_categories=84, device='cuda'):
        logger.info(f"Loading dataset from {X_file}")
        start = time.time()
        
        # Load data
        self.X = pl.read_parquet(X_file)
        self.y = pl.read_parquet(y_file).select(['item_cnt_day_winsor']).to_numpy().flatten().astype(np.float32)
        logger.info(f"Loaded Parquet files in {time.time() - start:.2f}s")
        
        if len(self.X) != len(self.y):
            raise ValueError(f"Mismatch between X ({len(self.X)}) and y ({len(self.y)}) rows")
        
        self.sequence_length = sequence_length
        self.num_shops = num_shops
        self.num_items = num_items
        self.num_categories = num_categories
        self.device = device
        
        self.numerical_cols = [
            'item_cnt_day_winsor', 'returns', 'item_price',
            'lag_sales_1', 'lag_sales_2', 'lag_sales_3',
            'lag_returns_1', 'lag_returns_2', 'lag_returns_3',
            'lag_price_1', 'lag_price_2', 'lag_price_3'
        ]
        self.categorical_cols = ['shop_id', 'item_id', 'item_category_id']
        
        # Validate and normalize numerical data
        numerical_data = self.X.select(self.numerical_cols).to_numpy()
        if np.isnan(numerical_data).any() or np.isnan(self.y).any():
            raise ValueError("NaN values in X or y")
        if np.isinf(numerical_data).any() or np.isinf(self.y).any():
            raise ValueError("Infinite values in X or y")
        
        # Clip and normalize numerical data
        numerical_data = np.clip(numerical_data, -1e5, 1e5)
        mean = numerical_data.mean(axis=0, keepdims=True)
        std = numerical_data.std(axis=0, keepdims=True) + 1e-6
        numerical_data = (numerical_data - mean) / std
        self.y = np.clip(self.y, -1e5, 1e5)
        
        # Clip categorical indices
        self.X = self.X.with_columns([
            pl.col('shop_id').clip_max(num_shops - 1),
            pl.col('item_id').clip_max(num_items - 1),
            pl.col('item_category_id').clip_max(num_categories - 1)
        ])
        
        # Preload data
        self.numerical = numerical_data.astype(np.float32)
        self.shop_ids = self.X['shop_id'].to_numpy().astype(np.int64)
        self.item_ids = self.X['item_id'].to_numpy().astype(np.int64)
        self.category_ids = self.X['item_category_id'].to_numpy().astype(np.int64)
        self.date_block_num = self.X['date_block_num'].to_numpy().astype(np.int32)
        
    def __len__(self):
        return len(self.X) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        try:
            start_idx = idx
            end_idx = idx + self.sequence_length
            if end_idx > len(self.X):
                raise IndexError("Index out of range")
            
            numerical = torch.tensor(self.numerical[start_idx:end_idx], dtype=torch.float32, device=self.device)
            shop_ids = torch.tensor(self.shop_ids[start_idx:end_idx], dtype=torch.int64, device=self.device)
            item_ids = torch.tensor(self.item_ids[start_idx:end_idx], dtype=torch.int64, device=self.device)
            category_ids = torch.tensor(self.category_ids[start_idx:end_idx], dtype=torch.int64, device=self.device)
            date_block_num = torch.tensor(self.date_block_num[start_idx:end_idx], dtype=torch.int32, device=self.device)
            target = torch.tensor(self.y[end_idx - 1], dtype=torch.float32, device=self.device)
            
            if torch.isnan(numerical).any() or torch.isnan(target).any():
                raise ValueError(f"NaN detected at index {idx}")
            
            identifiers = torch.tensor([
                int(self.shop_ids[end_idx - 1]), 
                int(self.item_ids[end_idx - 1]), 
                int(self.date_block_num[end_idx - 1])
            ], dtype=torch.int32, device=self.device)
            
            return {
                'numerical': numerical,
                'shop_ids': shop_ids,
                'item_ids': item_ids,
                'category_ids': category_ids,
                'target': target,
                'date_block_num': date_block_num[-1],
                'identifiers': identifiers
            }
        except Exception as e:
            logger.error(f"Error in __getitem__ at index {idx}: {str(e)}")
            raise

class FeatureAttention(nn.Module):
    def __init__(self, feature_dim, attention_dim=64):
        super(FeatureAttention, self).__init__()
        self.query = nn.Linear(feature_dim, attention_dim)
        self.key = nn.Linear(feature_dim, attention_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = 1 / (attention_dim ** 0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(feature_dim)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
    
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        scores = torch.bmm(query, key.transpose(1, 2)) * self.scale
        weights = self.softmax(scores)
        output = torch.bmm(weights, value)
        output = self.norm(output)
        return output, weights

class HALSTM(nn.Module):
    def __init__(self, num_shops, num_items, num_categories, embed_dim=16, numerical_dim=12, 
                 hidden_dim=128, num_layers=2, num_heads=4, dropout=0.2, forecast_horizon=1):
        super(HALSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        self.shop_embed = nn.Embedding(num_shops, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.category_embed = nn.Embedding(num_categories, embed_dim)
        nn.init.normal_(self.shop_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.item_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.category_embed.weight, mean=0.0, std=0.02)
        
        self.input_dim = embed_dim * 3 + numerical_dim
        self.feature_attention = FeatureAttention(self.input_dim)
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.lstm_norm = nn.LayerNorm(hidden_dim)
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc_shared = nn.Linear(hidden_dim, hidden_dim)
        self.fc_horizons = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(forecast_horizon)])
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, numerical, shop_ids, item_ids, category_ids, trace_mode=False):
        batch_size, seq_len, _ = numerical.size()
        shop_embed = self.shop_embed(shop_ids)
        item_embed = self.item_embed(item_ids)
        category_embed = self.category_embed(category_ids)
        
        x = torch.cat([numerical, shop_embed, item_embed, category_embed], dim=-1).contiguous()
        x, feature_weights = self.feature_attention(x)
        x = self.dropout(x)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device).contiguous()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device).contiguous()
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = self.lstm_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        mha_out, mha_weights = self.mha(lstm_out, lstm_out, lstm_out)
        mha_out = self.mha_norm(mha_out)
        mha_out = self.dropout(mha_out)
        
        combined = torch.cat([lstm_out[:, -1, :], mha_out[:, -1, :]], dim=-1)
        gate = self.sigmoid(self.gate(combined))
        fused = gate * lstm_out[:, -1, :] + (1 - gate) * mha_out[:, -1, :]
        
        shared = self.relu(self.fc_shared(fused))
        outputs = torch.cat([fc(shared).unsqueeze(1) for fc in self.fc_horizons], dim=1)
        outputs = outputs.squeeze(-1)
        
        if trace_mode:
            return outputs
        return outputs, {
            'feature_weights': feature_weights,
            'mha_weights': mha_weights,
            'fused_output': fused,
            'gate_weights': gate
        }

def collate_fn(batch):
    if not batch:
        logger.warning("Empty batch received")
        return {}
    
    numerical = torch.stack([item['numerical'] for item in batch])
    shop_ids = torch.stack([item['shop_ids'] for item in batch])
    item_ids = torch.stack([item['item_ids'] for item in batch])
    category_ids = torch.stack([item['category_ids'] for item in batch])
    target = torch.stack([item['target'] for item in batch])
    date_block_num = torch.stack([item['date_block_num'] for item in batch])
    identifiers = torch.stack([item['identifiers'] for item in batch])
    
    return {
        'numerical': numerical.contiguous(),
        'shop_ids': shop_ids.contiguous(),
        'item_ids': item_ids.contiguous(),
        'category_ids': category_ids.contiguous(),
        'target': target.contiguous(),
        'date_block_num': date_block_num.contiguous(),
        'identifiers': identifiers.contiguous()
    }