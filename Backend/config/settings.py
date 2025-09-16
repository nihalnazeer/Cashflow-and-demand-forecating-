from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import computed_field
from typing import Optional

class Settings(BaseSettings):
    """
    Centralized application settings with robust, absolute path definitions.
    """
    
    # --- BASE PATHS ---
    # This should point to the Backend directory since settings.py is in Backend/config/
    BASE_DIR: Path = Path(__file__).resolve().parent.parent  # Goes up to Backend/
    MODELS_DIR: Path = BASE_DIR / "models"
    RESULTS_DIR: Path = BASE_DIR / "results" 
    RAW_DATA_DIR: Path = BASE_DIR / "raw_data"
    
    # --- DATABASE ---
    POSTGRES_USER: str = "nihalnazeer"
    POSTGRES_PASSWORD: str = "your_password"
    POSTGRES_DB: str = "forecast"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    
    # --- REDIS ---
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Add lowercase alias for compatibility
    @property
    def redis_url(self) -> str:
        """Lowercase alias for REDIS_URL for compatibility"""
        return self.REDIS_URL
    
    @computed_field(return_type=str)
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    def get_model_path(self, filename: str) -> Path:
        """Get full path to model file"""
        return self.MODELS_DIR / filename
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in [self.MODELS_DIR, self.RESULTS_DIR, self.RAW_DATA_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields from environment

# Instantiate settings
settings = Settings()

# Ensure directories exist
settings.ensure_directories()

# Debug: Print paths for verification
if __name__ == "__main__":
    print(f"BASE_DIR: {settings.BASE_DIR}")
    print(f"MODELS_DIR: {settings.MODELS_DIR}")
    print(f"Model path exists: {settings.MODELS_DIR.exists()}")
    print(f"Looking for model at: {settings.get_model_path('best_ha_lstm.pth')}")
    print(f"Model file exists: {settings.get_model_path('best_ha_lstm.pth').exists()}")
    print(f"REDIS_URL: {settings.REDIS_URL}")
    print(f"redis_url: {settings.redis_url}")  # Test the property