# Checkpoint Loading Fix

## Problem Diagnosed

The application was failing to start with the following error:
```
RuntimeError: Error(s) in loading state_dict for MultimodalViralityPredictor:
    Missing key(s) in state_dict: "vision_encoder.resnet.*" [hundreds of missing keys]
    size mismatch for fusion.0.weight: copying a param with shape torch.Size([1024, 786])
    from checkpoint, the shape in current model is torch.Size([1024, 2834]).
```

## Root Cause

**Mismatch between checkpoint and inference configuration:**

- **Checkpoint** (trained model):
  - Trained with `use_vision=False`
  - Fusion input: 768 (BERT) + 18 (scalars) = **786 dimensions**
  - Only contains `text_encoder.*` and `fusion.*` weights

- **Inference code** (original):
  - Attempted to load with `use_vision=True`
  - Expected fusion input: 768 (BERT) + 2048 (ResNet-50) + 18 (scalars) = **2834 dimensions**
  - Expected both `text_encoder.*` and `vision_encoder.*` weights

## Solution Applied

### Changes to `app.py`

1. **Model Initialization** (line 115):
   ```python
   # Changed from use_vision=True to use_vision=False
   model = MultimodalViralityPredictor(
       num_scalar_features=18,
       freeze_encoders=True,
       fusion_hidden_dims=(1024, 256),
       dropout_rates=(0.3, 0.2),
       use_text=True,
       use_vision=False  # ← Fixed to match checkpoint
   )
   ```

2. **Prediction Endpoint** (line 272-276):
   ```python
   # Only preprocess image if model uses vision
   if model.use_vision:
       image_input = preprocess_image(req.thumbnail_url)
   else:
       image_input = None
   ```
   This avoids unnecessary image downloading and processing when vision is disabled.

3. **Model Info Endpoint** (line 318-338):
   ```python
   # Dynamically report enabled modalities
   "modalities": {
       "text": model.use_text,
       "vision": model.use_vision,
       "scalars": True
   }
   ```

## Implications

### Current Behavior
- ✅ Model loads successfully
- ✅ Predictions use **text + scalar features only**
- ✅ API still accepts `thumbnail_url` parameter (for compatibility)
- ✅ Image processing is skipped (faster inference)

### Performance Impact
- **Faster inference**: No image download or ResNet-50 processing
- **Lower memory**: ~25M fewer parameters (no ResNet-50 weights)
- **Reduced accuracy**: Vision features may improve virality prediction

### Future Improvements

If you want to use vision features:

1. **Retrain the model** with `use_vision=True`:
   ```python
   model = MultimodalViralityPredictor(
       num_scalar_features=18,
       use_text=True,
       use_vision=True  # Enable vision
   )
   ```

2. **Ensure training script** saves the full model including vision encoder weights

3. **Update checkpoint**: Replace `models/model_full.pt` with the new checkpoint

4. **Revert app.py changes**: Set `use_vision=True` in inference code

## Verification

To verify the fix works:

```bash
# Start the service
cd inference-service
python app.py

# Check health endpoint
curl http://localhost:7860/

# Check model info
curl http://localhost:7860/model/info

# Make a test prediction
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Amazing cat video!",
    "description": "Watch this cute cat do tricks",
    "thumbnail_url": "https://example.com/thumb.jpg",
    "view_count": 1000,
    "like_count": 100,
    "comment_count": 20,
    "duration_seconds": 30
  }'
```

Expected output:
- Health: `{"status": "healthy", "model_loaded": true}`
- Model info: `{"modalities": {"text": true, "vision": false, "scalars": true}}`
- Prediction: Valid viral score and velocity

## Technical Details

### Checkpoint Contents
```
Keys present:
- text_encoder.bert.*  (BERT weights)
- fusion.0.weight      [1024, 786]
- fusion.0.bias        [1024]
- fusion.2.weight      [256, 1024]
- fusion.2.bias        [256]
- classifier.weight    [2, 256]
- classifier.bias      [2]
- regressor.weight     [1, 256]
- regressor.bias       [1]

Keys missing:
- vision_encoder.*     (ALL ResNet-50 weights)
```

### Architecture
```
Input: Title + Description + Scalar Features
  ↓
BERT Encoder (768-dim)
  ↓
Concatenate with Scalars (18-dim)
  ↓
Fusion Layer (786 → 1024 → 256)
  ↓
├─ Classifier (256 → 2)    # Viral/Not Viral
└─ Regressor (256 → 1)     # Velocity Score
```

---

**Fixed by**: Claude Code
**Date**: 2025-12-04
**Issue**: Model checkpoint/inference configuration mismatch
