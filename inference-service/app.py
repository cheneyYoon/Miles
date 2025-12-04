"""
Miles Inference API
FastAPI service for multimodal virality prediction
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
import time
import logging
from typing import Optional
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertTokenizer
from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms as transforms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Miles Inference API",
    description="Multimodal virality prediction for short-form videos",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
tokenizer = None
device = None
image_transform = None


class PredictionRequest(BaseModel):
    title: str
    description: Optional[str] = ""
    thumbnail_url: str
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    duration_seconds: int = 30


class PredictionResponse(BaseModel):
    viral_score: float
    predicted_velocity: float
    confidence: float
    processing_time_ms: int
    model_version: str = "miles-v1.0"


def load_model():
    """Load the trained PyTorch model"""
    global model, tokenizer, device, image_transform

    # Import model architecture
    from src.models import MultimodalViralityPredictor

    logger.info("Loading model...")

    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logger.info("Tokenizer loaded")

    # Load model checkpoint
    model_path = "models/model_full.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    logger.info(f"Checkpoint loaded, type: {type(checkpoint)}")

    # Check if checkpoint is a dict or the model itself
    if isinstance(checkpoint, dict):
        # It's a checkpoint dict, instantiate model and load state dict
        logger.info("Loading from checkpoint dict")
        model = MultimodalViralityPredictor(
            num_scalar_features=18,  # From phase1_results.json
            freeze_encoders=True,
            fusion_hidden_dims=(1024, 256),
            dropout_rates=(0.3, 0.2),
            use_text=True,
            use_vision=True
        )

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the entire dict is the state dict
            model.load_state_dict(checkpoint)

        logger.info("State dict loaded successfully")
    else:
        # It's already a model object
        model = checkpoint
        logger.info("Model object loaded directly")

    model.to(device)
    model.eval()
    logger.info(f"Model ready on {device}")

    # Image preprocessing (same as training)
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    logger.info("✓ Model initialization complete")


def preprocess_text(title: str, description: str) -> dict:
    """Tokenize text for BERT"""
    text = f"{title} [SEP] {description}" if description else title

    encoded = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    return {
        'input_ids': encoded['input_ids'].to(device),
        'attention_mask': encoded['attention_mask'].to(device)
    }


def preprocess_image(url: str) -> torch.Tensor:
    """Download and preprocess thumbnail"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        tensor = image_transform(image)
        return tensor.unsqueeze(0).to(device)
    except Exception as e:
        logger.warning(f"Image preprocessing error: {e}. Using blank image.")
        # Return blank image as fallback
        return torch.zeros(1, 3, 224, 224).to(device)


def compute_scalars(req: PredictionRequest) -> torch.Tensor:
    """
    Extract scalar features (must match training setup)
    Based on 18 features from phase1_results.json
    """
    # Normalize features (using simple scaling)
    view_count_norm = min(req.view_count / 1_000_000, 10.0)  # Cap at 10M
    like_count_norm = min(req.like_count / 100_000, 10.0)
    comment_count_norm = min(req.comment_count / 10_000, 10.0)
    duration_norm = req.duration_seconds / 60.0

    # Compute engagement rates
    like_rate = req.like_count / max(req.view_count, 1)
    comment_rate = req.comment_count / max(req.view_count, 1)

    # Text features
    title_word_count = len(req.title.split())
    desc_word_count = len((req.description or "").split())
    title_length = len(req.title)

    # Binary features
    is_short = 1.0 if req.duration_seconds < 30 else 0.0
    is_popular = 1.0 if req.view_count > 100_000 else 0.0
    has_description = 1.0 if req.description else 0.0

    # Additional features to match 18-dim
    has_numbers = 1.0 if any(char.isdigit() for char in req.title) else 0.0
    has_question = 1.0 if '?' in req.title else 0.0
    has_exclamation = 1.0 if '!' in req.title else 0.0
    all_caps_words = sum(1 for word in req.title.split() if word.isupper() and len(word) > 1)
    engagement_score = (like_rate * 0.7 + comment_rate * 0.3) * 100

    features = torch.tensor([[
        view_count_norm,
        like_count_norm,
        comment_count_norm,
        duration_norm,
        like_rate,
        comment_rate,
        title_word_count,
        desc_word_count,
        title_length,
        is_short,
        is_popular,
        has_description,
        has_numbers,
        has_question,
        has_exclamation,
        all_caps_words,
        engagement_score,
        0.0  # Placeholder for 18th feature
    ]], dtype=torch.float32).to(device)

    return features


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "Miles v1.0",
        "device": str(device),
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    """
    Run inference on a video candidate

    Returns:
        viral_score: Probability of going viral [0-1]
        predicted_velocity: Expected view velocity
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        with torch.no_grad():
            # Preprocess inputs
            text_inputs = preprocess_text(req.title, req.description or "")
            image_input = preprocess_image(req.thumbnail_url)
            scalar_input = compute_scalars(req)

            # Run inference
            classification_logits, velocity_pred = model(
                text_input=text_inputs,
                image_input=image_input,
                scalar_features=scalar_input
            )

            # Convert logits to probability
            probs = F.softmax(classification_logits, dim=1)
            viral_prob = probs[0][1].item()  # Probability of positive class

            # Get velocity prediction
            velocity = velocity_pred[0][0].item()

            # Calculate confidence (distance from decision boundary)
            confidence = abs(viral_prob - 0.5) * 2

            processing_time = int((time.time() - start_time) * 1000)

            return PredictionResponse(
                viral_score=round(viral_prob, 4),
                predicted_velocity=round(max(0, velocity), 4),
                confidence=round(confidence, 2),
                processing_time_ms=processing_time
            )

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/model/info")
def model_info():
    """Get model information"""
    if model is None:
        return {"error": "Model not loaded"}

    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": param_count,
        "trainable_parameters": trainable_params,
        "device": str(device),
        "model_type": "MultimodalViralityPredictor",
        "components": {
            "text_encoder": "BERT (bert-base-uncased)",
            "vision_encoder": "ResNet-50",
            "fusion": "3-layer MLP"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
