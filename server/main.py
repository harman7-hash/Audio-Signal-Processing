# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException  # pyright: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import io
import os
from typing import List, Any
from time import perf_counter
import json
# Initialize FastAPI app
app = FastAPI(title="ML Model API", version="1.0.0")

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your ML model (defer heavy imports so the API can start without ML deps)
try:
    # Heavy imports only when attempting to load the model
    from tensorflow.keras.models import load_model  # type: ignore
    import pickle  # type: ignore
    import numpy as np  # type: ignore
    import librosa  # type: ignore

    model = load_model("model/anomaly_detector.h5")
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

# Feature extraction function (from your baseline code)
def extract_features(audio_data: Any, sr: int, n_mels: int = 64, frames: int = 5, 
                   n_fft: int = 1024, hop_length: int = 512, power: float = 2.0) -> Any:
    """Extract features from audio data"""
    # Calculate melspectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power
    )
    
    # Convert to log mel energy
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + np.finfo(float).eps)
    
    # Calculate total vector size
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1
    
    if vectorarray_size < 1:
        return np.empty((0, n_mels * frames), float)
    
    # Generate feature vectors
    dims = n_mels * frames
    vectorarray = np.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T
    
    return vectorarray

# API Routes
@app.get("/")
async def root():
    return {"message": "ML Model API is running", "status": "healthy"}

@app.get("/api/health")
async def health_check():
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "version": "1.0.0"
    }

@app.post("/api/predict")
async def predict_anomaly(file: UploadFile = File(...)):
    """
    Predict anomaly from audio file
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        t0 = perf_counter()
        # Read audio file
        contents = await file.read()
        audio_file = io.BytesIO(contents)
        
        # Load audio using librosa
        audio_data, sr = librosa.load(audio_file, sr=None, mono=True)
        
        # Extract features
        features = extract_features(
            audio_data=audio_data,
            sr=sr,
            n_mels=64,
            frames=5,
            n_fft=1024,
            hop_length=512,
            power=2.0
        )
        
        if features.shape[0] == 0:
            raise HTTPException(status_code=400, detail="Audio file too short for analysis")
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        predictions = model.predict(features_scaled)
        
        # Calculate reconstruction error (for anomaly detection)
        reconstruction_error = np.mean(np.square(features_scaled - predictions), axis=1)
        avg_error = np.mean(reconstruction_error)
        
        # Determine if anomalous based on threshold vs error
        threshold = 0.1  # default/model threshold; adjust as needed
        is_anomalous = avg_error > threshold
        confidence = 1.0 / (1.0 + np.exp(-avg_error))  # Simple confidence score
        processing_time_ms = int((perf_counter() - t0) * 1000)
        
        return {
            "filename": file.filename,
            "is_anomalous": bool(is_anomalous),
            "confidence": float(confidence),
            "reconstruction_error": float(avg_error),
            "error": float(avg_error),
            "threshold": float(threshold),
            "analysis_frames": int(features.shape[0]),
            "sample_rate": int(sr),
            "processing_time_ms": processing_time_ms
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/api/batch-predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Batch prediction for multiple audio files
    """
    results = []
    
    for file in files:
        try:
            # Reuse the single prediction logic
            result = await predict_anomaly(file)
            results.append(result)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "is_anomalous": None,
                "confidence": 0.0
            })
    
    return {"results": results}

# Serve static files for a simple frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)