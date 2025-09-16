import pandas as pd
from pathlib import Path
import sys
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()  # looks for a .env file in the project root

# --- CONFIGURATION ---
SOURCE_CSV_PATH = Path("/Users/mohammednihal/Desktop/XAI/Cashflow-and-demand-forecating-/Backend/raw_data/monthly_shops.csv")

# Fetch credentials from .env
POSTGRES_USER = os.getenv("POSTGRES_USER", "nihalnazeer")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")  # must be set in .env
POSTGRES_DB = os.getenv("POSTGRES_DB", "forecast")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
TABLE_NAME = "historical_monthly_sales"


def run_etl():
    """
    A simple ETL script to populate the analytics database.
    This should be run once to set up your historical data.
    """
    print("--- Starting ETL process to populate the analytics database ---")

    # 1. EXTRACT: Load your historical data from the CSV.
    try:
        df = pd.read_csv(SOURCE_CSV_PATH)
        print(f"✓ Successfully loaded {len(df)} rows from {SOURCE_CSV_PATH}")
    except FileNotFoundError:
        print(f"FATAL ERROR: Source data file not found at {SOURCE_CSV_PATH}. Please check the path.")
        sys.exit(1)

    # 2. TRANSFORM: Prepare the data for the database.
    print("Transforming data for database...")
    df['date'] = pd.to_datetime(df['date'])
    df_to_load = df[[
        'date',
        'shop_id',
        'shop_name',
        'item_id',
        'item_name',
        'item_category_id',
        'item_category_name',
        'item_cnt_month',
        'monthly_revenue'
    ]].copy()

    # 3. LOAD: Push the data to PostgreSQL.
    try:
        print(f"Connecting to database and writing to table '{TABLE_NAME}'...")
        engine = create_engine(DATABASE_URL)
        
        df_to_load.to_sql(
            TABLE_NAME,
            engine,
            if_exists='replace',
            index=False,
            chunksize=1000
        )
        print("\n" + "="*50)
        print(f"✅ DATA UPLOAD COMPLETE! ✅")
        print(f"Successfully loaded data into the '{TABLE_NAME}' table in the '{engine.url.database}' database.")
        print("="*50 + "\n")
    except Exception as e:
        print(f"FATAL ERROR: Database loading failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_etl()
