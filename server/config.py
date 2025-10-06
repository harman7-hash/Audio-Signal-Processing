# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME = "ML Model API"
    VERSION = "1.0.0"
    
    # Model paths
    MODEL_PATH = os.getenv("MODEL_PATH", "model/anomaly_detector.h5")
    SCALER_PATH = os.getenv("SCALER_PATH", "model/scaler.pkl")
    
    # API settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # CORS origins
    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # Add your production frontend URL here
    ]

settings = Settings()