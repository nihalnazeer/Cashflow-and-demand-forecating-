import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import polars as pl
from tqdm import tqdm
from pathlib import Path
import os
import logging
from model import SalesDataset, FeatureAttention, HALSTM, collate_fn

# Set file descriptor limit
os.system('ulimit -n 4096')

# Minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# GPU Configuration
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, accum_steps=4):
    logger.info("Starting training")
    criterion = nn.MSELoss().to(device)
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(train_loader)//accum_steps, pct_start=0.1
    )
    
    best_val_loss = float('inf')
    output_dir = Path('/workspace/processed_data')
    output_dir.mkdir(exist_ok=True)
    
    patience = 5
    early_stop_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in progress:
            try:
                numerical = batch['numerical'].to(device, non_blocking=True)
                shop_ids = batch['shop_ids'].to(device, non_blocking=True)
                item_ids = batch['item_ids'].to(device, non_blocking=True)
                category_ids = batch['category_ids'].to(device, non_blocking=True)
                target = batch['target'].to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    output, _ = model(numerical, shop_ids, item_ids, category_ids)
                    loss = criterion(output[:, -1], target) / accum_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                
                train_loss += loss.item() * accum_steps
                
                if batch_idx % 100 == 0:
                    logger.info(f"Batch {batch_idx}, GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
                              f"Utilization: {torch.cuda.utilization()}%")
                
                progress.set_postfix({"batch_loss": f"{loss.item() * accum_steps:.6f}"})
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                try:
                    numerical = batch['numerical'].to(device, non_blocking=True)
                    shop_ids = batch['shop_ids'].to(device, non_blocking=True)
                    item_ids = batch['item_ids'].to(device, non_blocking=True)
                    category_ids = batch['category_ids'].to(device, non_blocking=True)
                    target = batch['target'].to(device, non_blocking=True)
                    
                    with torch.cuda.amp.autocast():
                        output, _ = model(numerical, shop_ids, item_ids, category_ids)
                        loss = criterion(output[:, -1], target)
                    
                    if torch.isnan(loss):
                        logger.error("NaN detected in validation loss")
                        raise ValueError("Validation loss is NaN")
                    
                    val_loss += loss.item()
                except Exception as e:
                    logger.error(f"Error in validation: {str(e)}")
                    continue
        
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'best_ha_lstm.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
        
        torch.cuda.empty_cache()
    
    logger.info(f"Training done. Best val loss: {best_val_loss:.6f}")
    model.load_state_dict(torch.load(output_dir / 'best_ha_lstm.pth'))
    return model

def predict(model, test_loader):
    logger.info("Predicting")
    model.eval()
    predictions = []
    identifiers = []
    fused_outputs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            try:
                numerical = batch['numerical'].to(device, non_blocking=True)
                shop_ids = batch['shop_ids'].to(device, non_blocking=True)
                item_ids = batch['item_ids'].to(device, non_blocking=True)
                category_ids = batch['category_ids'].to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    output, attention_dict = model(numerical, shop_ids, item_ids, category_ids)
                    preds = output.cpu().numpy()
                    fused = attention_dict['fused_output'].cpu().numpy()
                
                predictions.append(preds)
                identifiers.append(batch['identifiers'].cpu().numpy())
                fused_outputs.append(fused)
            except Exception as e:
                logger.error(f"Error in prediction: {str(e)}")
                continue
            
            torch.cuda.empty_cache()
    
    if not predictions or not identifiers or not fused_outputs:
        logger.error("No predictions, identifiers, or fused outputs collected")
        return pd.DataFrame()
        
    predictions = np.concatenate(predictions, axis=0)
    identifiers = np.concatenate(identifiers, axis=0)
    fused_outputs = np.concatenate(fused_outputs, axis=0)
    
    pred_df = pd.DataFrame({
        'shop_id': identifiers[:, 0],
        'item_id': identifiers[:, 1],
        'date_block_num': identifiers[:, 2]
    })
    for h in range(predictions.shape[1]):
        pred_df[f'forecast_h{h+1}'] = predictions[:, h]
    
    fused_df = pd.DataFrame(fused_outputs, columns=[f'fused_dim_{i}' for i in range(fused_outputs.shape[1])])
    fused_df[['shop_id', 'item_id', 'date_block_num']] = identifiers
    
    output_dir = Path('/workspace/results')
    output_dir.mkdir(exist_ok=True)
    pred_df.to_csv(output_dir / 'predictions.csv', index=False)
    fused_df.to_csv(output_dir / 'fused_outputs.csv', index=False)
    logger.info(f"Predictions saved to {output_dir / 'predictions.csv'}")
    logger.info(f"Fused outputs saved to {output_dir / 'fused_outputs.csv'}")
    
    return pred_df

def visualize_results(pred_df, true_df=None, output_dir='/workspace/results'):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Saving plots to {output_dir}")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    
    if pred_df.empty:
        logger.error("Empty prediction dataframe")
        return None
    
    forecast_cols = [col for col in pred_df.columns if 'forecast' in col]
    if not forecast_cols:
        logger.error("No forecast columns found")
        return None
        
    for h in range(1, len(forecast_cols) + 1):
        sns.kdeplot(pred_df[f'forecast_h{h}'], label=f'Horizon {h}')
    
    plt.title('Prediction Distribution')
    plt.xlabel('Predicted Sales')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_distribution.png', dpi=300)
    plt.close()
    
    if true_df is not None and not true_df.empty:
        try:
            merged_df = pred_df.merge(true_df, on=['shop_id', 'item_id', 'date_block_num'], how='inner')
            if not merged_df.empty:
                merged_df['error'] = merged_df['forecast_h1'] - merged_df['item_cnt_day_winsor']
                
                plt.figure(figsize=(12, 8))
                sns.histplot(merged_df['error'], kde=True, bins=50)
                plt.title('Error Distribution')
                plt.xlabel('Prediction Error')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(output_dir / 'error_distribution.png', dpi=300)
                plt.close()
                
                plt.figure(figsize=(12, 8))
                plt.scatter(merged_df['item_cnt_day_winsor'], merged_df['forecast_h1'], alpha=0.5)
                plt.plot([0, merged_df['item_cnt_day_winsor'].max()], 
                         [0, merged_df['item_cnt_day_winsor'].max()], 'r--')
                plt.title('Predicted vs Actual')
                plt.xlabel('Actual Sales')
                plt.ylabel('Predicted Sales')
                plt.tight_layout()
                plt.savefig(output_dir / 'predicted_vs_actual.png', dpi=300)
                plt.close()
                
                mae = merged_df['error'].abs().mean()
                rmse = (merged_df['error'] ** 2).mean() ** 0.5
                logger.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
                
                return mae, rmse
            else:
                logger.warning("Merged dataframe is empty")
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")
    
    return None

def main():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    data_dir = Path('/workspace/processed_data')
    batch_size = 16384  # Increased for RTX A6000
    num_workers = 2
    prefetch_factor = 2
    num_shops = 60
    num_items = 22170
    num_categories = 84
    sequence_length = 12
    num_epochs = 50
    lr = 0.001
    accum_steps = 4
    
    logger.info("Loading datasets...")
    required_files = [
        'X_train_processed.parquet', 'y_train_processed.parquet',
        'X_val_processed.parquet', 'y_val_processed.parquet',
        'X_test_processed.parquet', 'y_test_processed.parquet'
    ]
    
    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")
    
    try:
        train_dataset = SalesDataset(
            data_dir / 'X_train_processed.parquet',
            data_dir / 'y_train_processed.parquet',
            sequence_length=sequence_length,
            num_shops=num_shops,
            num_items=num_items,
            num_categories=num_categories,
            device=device
        )
        val_dataset = SalesDataset(
            data_dir / 'X_val_processed.parquet',
            data_dir / 'y_val_processed.parquet',
            sequence_length=sequence_length,
            num_shops=num_shops,
            num_items=num_items,
            num_categories=num_categories,
            device=device
        )
        test_dataset = SalesDataset(
            data_dir / 'X_test_processed.parquet',
            data_dir / 'y_test_processed.parquet',
            sequence_length=sequence_length,
            num_shops=num_shops,
            num_items=num_items,
            num_categories=num_categories,
            device=device
        )
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Val dataset size: {len(val_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
        
    except Exception as e:
        logger.error(f"Error creating datasets: {str(e)}")
        raise
    
    try:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
            collate_fn=collate_fn,
            timeout=180
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
            collate_fn=collate_fn,
            timeout=180
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
            collate_fn=collate_fn,
            timeout=180
        )
        
        logger.info("DataLoaders created successfully")
        
    except Exception as e:
        logger.error(f"Error creating DataLoaders: {str(e)}")
        logger.warning("Falling back to num_workers=0")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
            collate_fn=collate_fn,
            timeout=180
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
            collate_fn=collate_fn,
            timeout=180
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
            collate_fn=collate_fn,
            timeout=180
        )
    
    try:
        model = HALSTM(
            num_shops=num_shops,
            num_items=num_items,
            num_categories=num_categories,
            embed_dim=16,
            numerical_dim=12,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            dropout=0.2,
            forecast_horizon=1
        ).to(device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        model = train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, accum_steps=accum_steps)
        
        predictions = predict(model, test_loader)
        
        y_test = pl.read_parquet(data_dir / 'y_test_processed.parquet')
        x_test_identifiers = pl.read_parquet(data_dir / 'X_test_processed.parquet').select(['shop_id', 'item_id', 'date_block_num'])
        true_df = pl.concat([x_test_identifiers, y_test], how='horizontal').to_pandas()
        
        visualize_results(predictions, true_df)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        logger.info("Starting program")
        main()
        logger.info("Program completed successfully")
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)