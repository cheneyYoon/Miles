---
title: Miles Inference API
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# Miles: Viral Video Prediction API

AI-powered multimodal inference service for predicting short-form video virality.

## 🎯 Overview

Miles uses a **deep learning model** combining:
- **BERT** (text encoder) - Analyzes titles and descriptions
- **Engagement metrics** - View counts, likes, comments, duration, etc.

**Note**: Current model version uses text + scalar features only. Vision encoder (ResNet-50) support is available but not enabled in this deployment.

**Performance**:
- AUROC: **0.855** (target: 0.75)
- Velocity MAE: **0.031** (target: 0.3)
- Trained on **9,542 videos**

## 🚀 API Endpoints

### `GET /`
Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "model": "Miles v1.0",
  "device": "cpu",
  "model_loaded": true
}
```

### `POST /predict`
Get virality prediction for a video

**Request**:
```json
{
  "title": "10 Python Tips Every Developer Should Know",
  "description": "Quick tutorial on Python best practices",
  "thumbnail_url": "https://i.ytimg.com/vi/abc123/maxresdefault.jpg",
  "view_count": 150000,
  "like_count": 5000,
  "comment_count": 300,
  "duration_seconds": 45
}
```

**Response**:
```json
{
  "viral_score": 0.8234,
  "predicted_velocity": 1247.32,
  "confidence": 0.85,
  "processing_time_ms": 342,
  "model_version": "miles-v1.0"
}
```

### `GET /model/info`
Get model architecture details

## 📊 Model Architecture

```
Input (Video Metadata)
├── Text (Title + Description)
│   └── BERT-base-uncased (768-dim)
└── Scalars (Engagement metrics)
    └── 18 features (views, likes, comments, duration, etc.)

    ↓ Concatenate (786-dim)
    ↓ Fusion MLP (786 → 1024 → 256)

Output
├── Classification: Viral or Not (binary)
└── Regression: View Velocity (float)
```

**Current Configuration**: Text + Scalars only (Vision encoder available but disabled)

## 🔧 Usage Example

### Python
```python
import requests

response = requests.post(
    "https://cheneyyoon-miles-inference.hf.space/predict",
    json={
        "title": "Amazing AI Breakthrough!",
        "description": "New research changes everything",
        "thumbnail_url": "https://example.com/thumb.jpg",
        "view_count": 50000,
        "like_count": 2000,
        "comment_count": 150,
        "duration_seconds": 30
    }
)

result = response.json()
print(f"Viral Score: {result['viral_score']:.2%}")
```

### cURL
```bash
curl -X POST https://cheneyyoon-miles-inference.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Amazing AI Breakthrough!",
    "thumbnail_url": "https://example.com/thumb.jpg",
    "view_count": 50000,
    "like_count": 2000,
    "comment_count": 150,
    "duration_seconds": 30
  }'
```

## 📚 Academic Context

**Course**: APS360 (Applied Fundamentals of Deep Learning)
**Institution**: University of Toronto
**Project**: Multimodal virality prediction for content creators

## 🛠️ Technical Stack

- **Framework**: FastAPI
- **ML Framework**: PyTorch 2.x
- **Text Model**: HuggingFace Transformers (BERT-base-uncased)
- **Deployment**: HuggingFace Spaces (Docker)
- **Container**: Python 3.11, FastAPI, PyTorch 2.x

## 📈 Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| AUROC | 0.855 | 0.75 | ✅ Exceeded |
| Velocity MAE | 0.031 | 0.3 | ✅ Exceeded |
| Inference Time | ~300ms | <1s | ✅ Met |

## 🔗 Related

- [Full Project Repository](https://github.com/cheneyyoon/Miles)
- [Frontend Demo](https://miles-mvp.vercel.app) _(coming soon)_

## 📄 License

MIT License - See LICENSE file for details

---

Built with ❤️ by Cheney Yoon | University of Toronto APS360
