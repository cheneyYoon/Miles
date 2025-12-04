# HuggingFace Spaces Deployment Guide

## Prerequisites

1. **HuggingFace Account**: You already have one (username: cheneyyoon)
2. **Git**: Installed on your system
3. **HuggingFace CLI** (optional but recommended)

---

## Step 1: Create Access Token

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Fill in:
   - **Name**: `miles-deployment`
   - **Type**: Select **"Write"**
4. Click **"Generate"**
5. **Copy the token** (starts with `hf_...`) - you'll need it in Step 3

⚠️ **Save this token securely!** You won't be able to see it again.

---

## Step 2: Create New Space

1. Go to: https://huggingface.co/new-space
2. Fill in:
   - **Owner**: `cheneyyoon`
   - **Space name**: `miles-inference`
   - **License**: `MIT`
   - **Select SDK**: Choose **"Docker"**
   - **Space hardware**: Select **"CPU basic"** (free)
   - **Visibility**: **Public** (required for free tier)
3. Click **"Create Space"**

You'll be redirected to your space: `https://huggingface.co/spaces/cheneyyoon/miles-inference`

---

## Step 3: Deploy to HuggingFace

### Option A: Using Git (Recommended)

Open terminal and run these commands:

```bash
# Navigate to inference-service directory
cd /Users/cheneyyoon/Desktop/03_UofT/APS360/Miles/inference-service

# Initialize git (if not already done)
git init

# Add HuggingFace remote
git remote add hf https://huggingface.co/spaces/cheneyyoon/miles-inference

# Configure git credential helper (so you only enter token once)
git config credential.helper store

# Add all files
git add .

# Create initial commit
git commit -m "Initial deployment: Miles Inference API v1.0"

# Push to HuggingFace
git push hf main
```

When prompted for credentials:
- **Username**: `cheneyyoon`
- **Password**: `hf_...` (paste your access token from Step 1)

---

### Option B: Using HuggingFace CLI (Alternative)

If you prefer using the CLI:

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login
huggingface-cli login
# Paste your token when prompted

# Upload space
cd /Users/cheneyyoon/Desktop/03_UofT/APS360/Miles/inference-service
huggingface-cli upload cheneyyoon/miles-inference . --repo-type=space
```

---

## Step 4: Monitor Build Progress

1. Go to your space: https://huggingface.co/spaces/cheneyyoon/miles-inference
2. Click the **"Building"** tab at the top
3. Watch the logs as Docker builds your container

**Expected timeline**:
- Docker build: ~5-10 minutes
- Model loading: ~1-2 minutes
- **Total**: ~10-15 minutes

---

## Step 5: Verify Deployment

Once status shows **"Running"** (green):

### Test Health Endpoint
```bash
curl https://cheneyyoon-miles-inference.hf.space/
```

**Expected response**:
```json
{
  "status": "healthy",
  "model": "Miles v1.0",
  "device": "cpu",
  "model_loaded": true
}
```

### Test Prediction Endpoint
```bash
curl -X POST https://cheneyyoon-miles-inference.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "10 Python Tips Every Developer Should Know",
    "thumbnail_url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
    "view_count": 150000,
    "like_count": 5000,
    "comment_count": 300,
    "duration_seconds": 45
  }'
```

**Expected response**:
```json
{
  "viral_score": 0.7234,
  "predicted_velocity": 1247.32,
  "confidence": 0.85,
  "processing_time_ms": 342,
  "model_version": "miles-v1.0"
}
```

---

## Step 6: Update Scraper Configuration

Update your scraper to use the HuggingFace URL:

```bash
# Edit scraper/.env
INFERENCE_API_URL=https://cheneyyoon-miles-inference.hf.space/predict
```

Then test:
```bash
cd /Users/cheneyyoon/Desktop/03_UofT/APS360/Miles/scraper
source venv/bin/activate
python trigger_inference.py
```

---

## Troubleshooting

### Build Failed
- Check the build logs in HuggingFace Space
- Common issues:
  - Model file too large (422MB should be OK)
  - Missing dependencies (check requirements.txt)
  - Python version mismatch (Dockerfile uses 3.11)

### Space is Sleeping
- Free tier spaces sleep after ~48 hours of inactivity
- They wake up on first request (takes ~30 seconds)
- Add a "Waking up..." message in your frontend

### 503 Service Unavailable
- Space is still building
- Wait 2-3 more minutes and try again

### Authentication Failed (git push)
- Make sure you're using your **access token**, not your HuggingFace password
- Token should start with `hf_`

---

## Space URL

Once deployed, your inference API will be available at:

🔗 **https://cheneyyoon-miles-inference.hf.space**

API Endpoints:
- Health: `https://cheneyyoon-miles-inference.hf.space/`
- Predict: `https://cheneyyoon-miles-inference.hf.space/predict`
- Model Info: `https://cheneyyoon-miles-inference.hf.space/model/info`

---

## Next Steps

After deployment:
1. ✅ Update `scraper/.env` with HF URL
2. ✅ Test end-to-end pipeline
3. ✅ Update `frontend/.env.local` if needed
4. ✅ Run full scraper → inference → database flow

---

**Good luck with deployment! Let me know if you run into any issues.**
