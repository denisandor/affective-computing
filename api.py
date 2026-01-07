"""
Simple FastAPI server for emotion detection and music recommendation.
Accepts audio and image file uploads, processes through models, returns predictions.
"""
import os
import logging
import tempfile
from typing import Optional

import torch
import torchvision.transforms as transforms
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import librosa
import cv2
import numpy as np
from PIL import Image
import io

from core.config import CONFIG, device
from audio.wav2vec2 import Wav2Vec2
from fusion.model import FusionModel
from facial.model import FacialEmotionNet
from transformers import Wav2Vec2Processor
from recommendation.recommender import generate_mood_shift_playlist, aggregate_sentiments
from recommendation.spotify_client import SpotifyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    load_models()
    
    # Initialize Spotify client if credentials are available
    global spotify_client
    try:
        client_id = os.environ.get('SPOTIFY_CLIENT_ID')
        client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
        if client_id and client_secret:
            spotify_client = SpotifyClient(client_id=client_id, client_secret=client_secret)
            logger.info("Spotify client initialized - will use Spotify API for recommendations")
        else:
            logger.info("Spotify credentials not set - will use local CSV database")
            # Try to use local CSV database
            csv_path = os.path.join("recommendation", "dataset.csv")
            if os.path.exists(csv_path):
                import pandas as pd
                df = pd.read_csv(csv_path)
                logger.info(f"Local CSV database found: {csv_path} ({len(df):,}+ tracks)")
            else:
                logger.warning(f"Local CSV database not found: {csv_path}")
    except Exception as e:
        logger.warning(f"Error initializing Spotify client: {e}")
    
    yield
    # Cleanup code can go here if needed

app = FastAPI(title="Emotion-Aware Music Player API", lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
audio_model = None
facial_model = None
fusion_model = None
audio_processor = None
emotion_labels = None
spotify_client = None

# Emotion mapping from IEMOCAP labels to readable names
EMOTION_MAP = {
    "ang": "angry",
    "fru": "frustrated", 
    "neu": "neutral",
    "sad": "sad",
    "exc": "excited",
    "hap": "happy"
}


def load_models():
    """Load trained models and processor."""
    global audio_model, facial_model, fusion_model, audio_processor, emotion_labels
    
    try:
        CONFIG.load_config("config.yaml")
        emotion_labels = CONFIG.dataset_emotions()
        num_classes = len(emotion_labels)
        
        logger.info("Loading audio model...")
        audio_model = Wav2Vec2(num_classes=num_classes)
        audio_checkpoint = os.path.join(CONFIG.saved_models_location(), "wav2vec2_state_dict1.pt")
        if os.path.exists(audio_checkpoint):
            state_dict = torch.load(audio_checkpoint, map_location=device)
            audio_model.load_state_dict(state_dict)
            logger.info(f"Loaded audio checkpoint: {audio_checkpoint}")
        else:
            logger.warning(f"Audio checkpoint not found: {audio_checkpoint}")
        
        audio_model.to(device)
        audio_model.eval()
        audio_model.freeze_feature_extractor()
        
        logger.info("Loading facial model...")
        facial_model = FacialEmotionNet(
            num_classes=num_classes,
            embedding_dim=128,
            pretrained=True,
            freeze_backbone=False,
        )
        facial_checkpoint = os.path.join(CONFIG.saved_models_location(), "facial_test_only.pt")
        if os.path.exists(facial_checkpoint):
            state_dict = torch.load(facial_checkpoint, map_location=device)
            facial_model.load_state_dict(state_dict)
            logger.info(f"Loaded facial checkpoint: {facial_checkpoint}")
        else:
            logger.warning(f"Facial checkpoint not found: {facial_checkpoint}")
        
        facial_model.to(device)
        facial_model.eval()
        
        logger.info("Loading fusion model...")
        # Fusion model was trained with face_dim=7, so we need to match that
        # We'll project the 128-dim facial embedding to 7-dim to match the checkpoint
        fusion_model = FusionModel(
            num_classes=num_classes,
            wav2vec2_model=audio_model,
            face_dim=7,  # Match the original training configuration
            freeze_audio=True,
            freeze_face=True,
        )
        fusion_checkpoint = os.path.join(CONFIG.saved_models_location(), "fusion_state_dict.pt")
        if os.path.exists(fusion_checkpoint):
            state_dict = torch.load(fusion_checkpoint, map_location=device)
            fusion_model.load_state_dict(state_dict)
            logger.info(f"Loaded fusion checkpoint: {fusion_checkpoint}")
        else:
            logger.warning(f"Fusion checkpoint not found: {fusion_checkpoint}")
        
        fusion_model.to(device)
        fusion_model.eval()
        
        logger.info("Loading audio processor...")
        audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        logger.info("Models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


def process_audio_file(audio_file: UploadFile) -> torch.Tensor:
    """Process uploaded audio file to tensor format."""
    try:
        # Save uploaded file temporarily - use generic suffix, librosa can handle various formats
        content = audio_file.file.read()
        if len(content) == 0:
            raise ValueError("Audio file is empty")
        
        # Determine file extension from content type or use generic
        suffix = ".webm"  # MediaRecorder typically creates WebM
        if audio_file.content_type:
            if "wav" in audio_file.content_type.lower():
                suffix = ".wav"
            elif "mp3" in audio_file.content_type.lower():
                suffix = ".mp3"
            elif "ogg" in audio_file.content_type.lower():
                suffix = ".ogg"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Load and process audio - librosa can handle WebM if ffmpeg is available
        try:
            waveform, sample_rate = librosa.load(tmp_path, sr=16000, duration=3.0)
        except Exception as e:
            # If librosa fails, try without duration limit or with different approach
            logger.warning(f"First attempt failed: {e}, trying alternative...")
            waveform, sample_rate = librosa.load(tmp_path, sr=16000)
        
        os.unlink(tmp_path)  # Clean up temp file
        
        if len(waveform) == 0:
            raise ValueError("Audio file contains no audio data")
        
        # Convert to tensor - librosa returns 1D numpy array
        waveform_tensor = torch.tensor(waveform)  # Keep as 1D, processor will handle batching
        
        # Process with Wav2Vec2 processor
        processed = audio_processor(
            waveform_tensor,
            return_tensors="pt",
            sampling_rate=sample_rate,
        )
        input_features = processed.input_values.to(device)
        
        # Ensure correct shape: should be [batch, sequence] = [1, seq_len]
        if input_features.dim() != 2:
            # Fix shape if processor added extra dimensions
            input_features = input_features.squeeze()
            if input_features.dim() == 1:
                input_features = input_features.unsqueeze(0)
        
        return input_features
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        error_msg = str(e)
        if "No such file" in error_msg or "cannot identify" in error_msg.lower():
            error_msg = "Audio format not supported. Please use WAV, MP3, or ensure ffmpeg is installed for WebM support."
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {error_msg}")


def process_image_file(image_file: UploadFile) -> torch.Tensor:
    """Process uploaded image file to tensor format."""
    try:
        # Read image bytes
        image_bytes = image_file.file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize to 224x224
        img_resized = cv2.resize(img_array, (224, 224))
        
        # Normalize and convert to tensor
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor = transform(img_resized).unsqueeze(0)
        return tensor.to(device)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    load_models()
    yield
    # Cleanup code can go here if needed
    
    # Initialize Spotify client if credentials are available
    global spotify_client
    try:
        client_id = os.environ.get('SPOTIFY_CLIENT_ID')
        client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
        if client_id and client_secret:
            spotify_client = SpotifyClient(client_id=client_id, client_secret=client_secret)
            logger.info("Spotify client initialized - will use Spotify API for recommendations")
        else:
            logger.info("Spotify credentials not set - will use local CSV database")
    except Exception as e:
        logger.warning(f"Spotify client initialization failed: {e}")
        logger.info("Will fall back to local CSV database")
    
    # Check if CSV database is available
    csv_path = os.path.join("recommendation", "dataset.csv")
    if os.path.exists(csv_path):
        logger.info(f"Local CSV database found: {csv_path} (114k+ tracks)")
    else:
        logger.warning(f"Local CSV database not found: {csv_path}")
        logger.info("Will use hardcoded catalog as fallback")


@app.get("/")
async def root():
    """Serve frontend or health check."""
    frontend_path = os.path.join("frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"status": "ok", "message": "Emotion-Aware Music Player API"}

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "message": "Emotion-Aware Music Player API"}


@app.post("/api/detect-emotion")
async def detect_emotion(
    audio: UploadFile = File(...),
    image: UploadFile = File(...)
):
    """
    Detect emotion from audio and image files using fusion model.
    
    Parameters:
    - audio: Audio file (required)
    - image: Image file (required)
    
    Returns:
    - emotion: detected emotion label
    - confidence: confidence score
    - modality: "both" (fusion)
    - all_probabilities: probability distribution over all emotions
    """
    if audio_model is None or facial_model is None or fusion_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        with torch.no_grad():
            # Process audio and face separately with their respective models
            audio_tensor = process_audio_file(audio)
            face_tensor = process_image_file(image)
            
            # Get individual model predictions
            # Facial model prediction
            facial_logits, facial_emb = facial_model(face_tensor, return_features=True)
            facial_probs = torch.softmax(facial_logits, dim=1)
            facial_pred = torch.argmax(facial_probs, dim=1).item()
            facial_confidence = facial_probs[0][facial_pred].item()
            
            # Audio model prediction
            audio_output = audio_model(audio_tensor)
            audio_probs = audio_output
            audio_pred = torch.argmax(audio_probs, dim=1).item()
            audio_confidence = audio_probs[0][audio_pred].item()
            
            # Get audio embeddings from Wav2Vec2 (before final classification)
            # Extract the pooled hidden state (768 dim) and project it
            audio_wav2vec_outputs = audio_model.wav2vec2(audio_tensor)
            audio_last_hidden = audio_wav2vec_outputs.last_hidden_state
            audio_pooled = audio_last_hidden.mean(dim=1)  # [batch, 768]
            
            # Project audio using fusion model's audio_proj (768 -> 256)
            audio_emb = fusion_model.audio_proj(audio_pooled)  # [batch, 256]
            
            # Facial embedding from FacialEmotionNet is [batch, 128]
            # Fusion model was trained with face_dim=7, so we need to project 128 -> 7
            # Create a projection layer if it doesn't exist
            if not hasattr(fusion_model, '_facial_emb_proj'):
                fusion_model._facial_emb_proj = torch.nn.Linear(128, 7).to(device)
                # Initialize with small weights
                torch.nn.init.xavier_uniform_(fusion_model._facial_emb_proj.weight)
                torch.nn.init.zeros_(fusion_model._facial_emb_proj.bias)
            
            # Project facial embedding from 128 to 7 dimensions to match fusion model
            facial_emb_projected = fusion_model._facial_emb_proj(facial_emb)  # [batch, 7]
            
            # Concatenate embeddings: [batch, 256 + 7] = [batch, 263]
            fusion_in = torch.cat([audio_emb, facial_emb_projected], dim=1)
            
            # Pass through fusion model's MLP and classification head
            h = fusion_model.mlp(fusion_in)
            fusion_logits = fusion_model.cls_head(h)
            
            # Get fusion prediction
            fusion_probs = torch.softmax(fusion_logits, dim=1)
            fusion_pred = torch.argmax(fusion_probs, dim=1).item()
            fusion_confidence = fusion_probs[0][fusion_pred].item()
            
            detected_emotion = emotion_labels[fusion_pred]
            all_probs = {
                emotion_labels[i]: fusion_probs[0][i].item() 
                for i in range(len(emotion_labels))
            }
            confidence = fusion_confidence
        
        # Map to readable emotion names
        readable_emotion = EMOTION_MAP.get(detected_emotion, detected_emotion)
        
        # Debug logging
        logger.info(f"Facial model prediction - Emotion: {emotion_labels[facial_pred]}, Confidence: {facial_confidence:.4f}")
        logger.info(f"Audio model prediction - Emotion: {emotion_labels[audio_pred]}, Confidence: {audio_confidence:.4f}")
        logger.info(f"Fusion model prediction - Emotion: {detected_emotion}, Confidence: {confidence:.4f}")
        logger.info(f"All probabilities: {all_probs}")
        
        result = {
            "emotion": readable_emotion,
            "emotion_code": detected_emotion,
            "confidence": round(confidence, 4),
            "modality": "both",
            "all_probabilities": {k: round(v, 4) for k, v in all_probs.items()},
            "facial_emotion": EMOTION_MAP.get(emotion_labels[facial_pred], emotion_labels[facial_pred]),
            "facial_confidence": round(facial_confidence, 4),
            "audio_emotion": EMOTION_MAP.get(emotion_labels[audio_pred], emotion_labels[audio_pred]),
            "audio_confidence": round(audio_confidence, 4)
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in emotion detection: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.get("/api/recommend-music")
async def recommend_music(emotion: str):
    """
    Get music recommendations based on detected emotion.
    
    Parameters:
    - emotion: emotion label (e.g., "sad", "happy", "angry", "neutral")
    
    Returns:
    - playlist: list of recommended tracks
    """
    try:
        # Generate playlist based on emotion
        # Priority: Spotify API > Local CSV > Hardcoded Catalog
        csv_path = os.path.join("recommendation", "dataset.csv")
        playlist = generate_mood_shift_playlist(
            emotion.lower(),
            sp_client=spotify_client,
            local_csv_path=csv_path if os.path.exists(csv_path) else None
        )
        
        return {
            "emotion": emotion,
            "playlist": playlist,
            "track_count": len(playlist)
        }
        
    except Exception as e:
        logger.error(f"Error in music recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.post("/api/detect-and-recommend")
async def detect_and_recommend(
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    modality: Optional[str] = None
):
    """
    Combined endpoint: detect emotion and get music recommendations in one call.
    Accepts either audio, image, or both.
    """
    # Detect emotion
    emotion_result = await detect_emotion(audio, image, modality)
    
    # Get recommendations
    readable_emotion = emotion_result["emotion"]
    try:
        recommendations = await recommend_music(readable_emotion)
    except Exception as e:
        # Fallback if recommendation fails
        logger.warning(f"Recommendation failed in combined endpoint: {e}")
        recommendations = {"playlist": [], "track_count": 0}
    
    return {
        **emotion_result,
        "recommendations": recommendations
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

