"""
Quick validation script to test all modules before running in Colab.

Run this locally to catch issues early:
    python tests/test_modules_quick.py

This will validate:
- All imports work
- Data preprocessing functions
- Model architectures
- Training utilities
- Basic forward passes

Does NOT require:
- Actual dataset downloaded
- GPU available
- Long training runs
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

print("="*70)
print("Miles Project - Quick Module Validation")
print("="*70)
print()

# Track results
tests_passed = 0
tests_failed = 0
errors = []

def test_section(name):
    """Decorator for test sections"""
    def decorator(func):
        def wrapper():
            global tests_passed, tests_failed
            print(f"\n{'='*70}")
            print(f"Testing: {name}")
            print(f"{'='*70}")
            try:
                func()
                print(f"✅ {name} - PASSED")
                tests_passed += 1
                return True
            except Exception as e:
                print(f"❌ {name} - FAILED")
                print(f"Error: {e}")
                errors.append((name, str(e)))
                tests_failed += 1
                return False
        return wrapper
    return decorator


# ============================================================================
# TEST 1: IMPORTS
# ============================================================================

@test_section("Module Imports")
def test_imports():
    """Test that all modules can be imported"""
    print("Importing core libraries...")
    import numpy as np
    import pandas as pd
    import torch
    print(f"  ✓ NumPy {np.__version__}")
    print(f"  ✓ Pandas {pd.__version__}")
    print(f"  ✓ PyTorch {torch.__version__}")

    print("\nImporting data modules...")
    from data.download import download_dataset
    from data.preprocessing import TextPreprocessor, ImagePreprocessor
    from data.feature_engineering import EngagementFeatureEngineer, create_viral_labels
    from data.dataset import ViralShortsDataset, create_train_val_test_split
    print("  ✓ All data modules")

    print("\nImporting model modules...")
    from models.baseline import BaselineModel
    from models.text_encoder import BERTTextEncoder
    from models.vision_encoder import ResNetVisionEncoder
    from models.fusion_model import MultimodalViralityPredictor
    print("  ✓ All model modules")

    print("\nImporting training modules...")
    from training.utils import load_config, set_seed, get_device
    from training.evaluate import evaluate_model, compute_metrics
    from training.train import train_epoch
    print("  ✓ All training modules")


# ============================================================================
# TEST 2: DATA PREPROCESSING
# ============================================================================

@test_section("Data Preprocessing")
def test_preprocessing():
    """Test text and data preprocessing"""
    import pandas as pd
    from data.preprocessing import TextPreprocessor, preprocess_dataset

    # Test TextPreprocessor
    print("Testing TextPreprocessor...")
    text_prep = TextPreprocessor()

    test_text = "Check out this AMAZING video! https://youtu.be/test #viral #trending 🔥"
    cleaned = text_prep.clean_text(test_text)
    print(f"  Original: {test_text[:50]}...")
    print(f"  Cleaned: {cleaned[:50]}...")
    assert len(cleaned) > 0, "Cleaned text is empty"
    assert "http" not in cleaned, "URLs not removed"

    # Test tokenization
    tokens = text_prep.tokenize(test_text)
    assert 'input_ids' in tokens, "Missing input_ids"
    assert 'attention_mask' in tokens, "Missing attention_mask"
    assert tokens['input_ids'].shape[1] == 128, "Wrong sequence length"
    print(f"  ✓ Tokenization: shape {tokens['input_ids'].shape}")

    # Test dataset preprocessing
    print("\nTesting dataset preprocessing...")
    df_test = pd.DataFrame({
        'row_id': [1, 2, 3],
        'title': ['Video 1', 'Video 2', 'Video 3'],
        'views': [1000, 2000, 3000],
        'likes': [100, 200, 300],
        'language': ['en', 'en', 'en']
    })

    df_clean = preprocess_dataset(df_test, text_columns=['title'], deduplicate=False)
    assert len(df_clean) > 0, "Preprocessing removed all rows"
    assert 'title_clean' in df_clean.columns, "Missing cleaned title column"
    print(f"  ✓ Dataset preprocessing: {len(df_test)} → {len(df_clean)} rows")


# ============================================================================
# TEST 3: FEATURE ENGINEERING
# ============================================================================

@test_section("Feature Engineering")
def test_feature_engineering():
    """Test engagement feature engineering"""
    import pandas as pd
    import numpy as np
    from data.feature_engineering import (
        EngagementFeatureEngineer,
        create_viral_labels,
        extract_temporal_features
    )

    print("Testing EngagementFeatureEngineer...")
    engineer = EngagementFeatureEngineer()

    # Create test data
    df_test = pd.DataFrame({
        'video_id': [1, 2, 3, 4, 5],
        'views': [1000, 5000, 10000, 500, 50000],
        'likes': [100, 400, 800, 50, 4000],
        'comments': [10, 50, 100, 5, 500],
        'upload_date': pd.date_range(end=pd.Timestamp.now(), periods=5, freq='D')
    })

    # Test normalization
    df_norm = engineer.normalize_engagement_metrics(df_test)
    assert 'age_hours' in df_norm.columns, "Missing age_hours"
    assert 'views_per_hour' in df_norm.columns, "Missing views_per_hour"
    print(f"  ✓ Normalized metrics: {len(df_norm.columns)} columns")

    # Test velocity calculation
    df_norm['engagement_velocity'] = engineer.calculate_engagement_velocity(df_norm)
    assert df_norm['engagement_velocity'].notna().all(), "NaN in velocity"
    print(f"  ✓ Engagement velocity calculated")

    # Test viral labels
    labels, threshold = create_viral_labels(df_norm)
    assert len(labels) == len(df_norm), "Label count mismatch"
    assert labels.sum() > 0, "No viral videos labeled"
    print(f"  ✓ Viral labels: {labels.sum()}/{len(labels)} viral (threshold={threshold:.2f})")

    # Test temporal features
    df_temporal = extract_temporal_features(df_test)
    assert 'upload_day_of_week' in df_temporal.columns, "Missing day of week"
    assert 'is_weekend' in df_temporal.columns, "Missing weekend flag"
    print(f"  ✓ Temporal features extracted")


# ============================================================================
# TEST 4: PYTORCH DATASET
# ============================================================================

@test_section("PyTorch Dataset")
def test_pytorch_dataset():
    """Test PyTorch Dataset creation"""
    import pandas as pd
    import torch
    from data.dataset import ViralShortsDataset, create_train_val_test_split
    from torch.utils.data import DataLoader

    print("Creating test dataset...")
    df_test = pd.DataFrame({
        'video_id': range(100),
        'title': [f'Video {i}' for i in range(100)],
        'views_per_hour': torch.rand(100).numpy() * 1000,
        'likes_per_hour': torch.rand(100).numpy() * 100,
        'upload_day_of_week': torch.randint(0, 7, (100,)).numpy(),
        'upload_hour': torch.randint(0, 24, (100,)).numpy(),
        'is_weekend': torch.randint(0, 2, (100,)).numpy(),
        'engagement_velocity': torch.rand(100).numpy() * 100,
        'is_viral': torch.randint(0, 2, (100,)).numpy(),
    })

    # Test train/val/test split
    print("Testing data splits...")
    train_df, val_df, test_df = create_train_val_test_split(
        df_test, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    assert len(train_df) + len(val_df) + len(test_df) == len(df_test), "Split size mismatch"
    print(f"  ✓ Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Test dataset creation (text-only)
    print("Testing ViralShortsDataset...")
    dataset = ViralShortsDataset(
        train_df,
        text_column='title',
        use_images=False
    )

    assert len(dataset) == len(train_df), "Dataset size mismatch"
    print(f"  ✓ Dataset created: {len(dataset)} samples")

    # Test __getitem__
    sample = dataset[0]
    assert 'text' in sample, "Missing text"
    assert 'scalars' in sample, "Missing scalars"
    assert 'label' in sample, "Missing label"
    assert 'velocity' in sample, "Missing velocity"
    print(f"  ✓ Sample shape: text={sample['text']['input_ids'].shape}, scalars={sample['scalars'].shape}")

    # Test DataLoader
    print("Testing DataLoader...")
    from data.dataset import collate_multimodal_batch
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_multimodal_batch)
    batch = next(iter(loader))
    assert batch['text']['input_ids'].shape[0] == 4, "Batch size mismatch"
    print(f"  ✓ DataLoader batch: {batch['text']['input_ids'].shape}")


# ============================================================================
# TEST 5: BASELINE MODEL
# ============================================================================

@test_section("Baseline Model")
def test_baseline_model():
    """Test baseline logistic regression model"""
    import pandas as pd
    from models.baseline import BaselineModel

    print("Testing BaselineModel...")

    # Create training data
    df_train = pd.DataFrame({
        'title': [f'Title {i}' for i in range(50)],
        'is_viral': [i % 2 for i in range(50)]
    })

    df_test = pd.DataFrame({
        'title': [f'Test title {i}' for i in range(10)],
        'is_viral': [i % 2 for i in range(10)]
    })

    # Train model
    model = BaselineModel(max_features=100, max_iter=100)
    model.fit(df_train, label_column='is_viral', text_columns=['title'])
    print(f"  ✓ Model trained")

    # Test prediction
    predictions = model.predict(df_test, text_columns=['title'])
    assert len(predictions) == len(df_test), "Prediction count mismatch"
    print(f"  ✓ Predictions: {predictions[:5]}")

    # Test probability prediction
    probs = model.predict_proba(df_test, text_columns=['title'])
    assert probs.shape == (len(df_test), 2), "Probability shape mismatch"
    print(f"  ✓ Probabilities shape: {probs.shape}")


# ============================================================================
# TEST 6: TEXT ENCODER
# ============================================================================

@test_section("Text Encoder (BERT)")
def test_text_encoder():
    """Test BERT text encoder"""
    import torch
    from models.text_encoder import BERTTextEncoder

    print("Testing BERTTextEncoder...")
    print("  (This may take a minute to download BERT weights...)")

    encoder = BERTTextEncoder(freeze=True)
    params = encoder.count_parameters()
    print(f"  ✓ Model initialized: {params['total']:,} total params, {params['trainable']:,} trainable")

    # Test forward pass
    batch_size = 2
    seq_length = 128
    dummy_input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length)

    embeddings = encoder(dummy_input_ids, dummy_attention_mask)
    assert embeddings.shape == (batch_size, 768), f"Wrong output shape: {embeddings.shape}"
    print(f"  ✓ Forward pass: {embeddings.shape}")

    # Test unfreezing
    encoder.unfreeze_bert(num_layers=2)
    params_unfrozen = encoder.count_parameters()
    assert params_unfrozen['trainable'] > params['trainable'], "Unfreezing didn't work"
    print(f"  ✓ Unfroze 2 layers: {params_unfrozen['trainable']:,} trainable params")


# ============================================================================
# TEST 7: VISION ENCODER
# ============================================================================

@test_section("Vision Encoder (ResNet)")
def test_vision_encoder():
    """Test ResNet vision encoder"""
    import torch
    from models.vision_encoder import ResNetVisionEncoder

    print("Testing ResNetVisionEncoder...")
    print("  (This may take a minute to download ResNet weights...)")

    encoder = ResNetVisionEncoder(freeze=True, pretrained=True)
    params = encoder.count_parameters()
    print(f"  ✓ Model initialized: {params['total']:,} total params, {params['trainable']:,} trainable")

    # Test forward pass
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)

    features = encoder(dummy_images)
    assert features.shape == (batch_size, 2048), f"Wrong output shape: {features.shape}"
    print(f"  ✓ Forward pass: {features.shape}")


# ============================================================================
# TEST 8: MULTIMODAL FUSION MODEL
# ============================================================================

@test_section("Multimodal Fusion Model")
def test_fusion_model():
    """Test multimodal fusion model"""
    import torch
    from models.fusion_model import MultimodalViralityPredictor

    print("Testing MultimodalViralityPredictor...")
    print("  (This will download both BERT and ResNet weights...)")

    # Text-only model for faster testing
    model = MultimodalViralityPredictor(
        num_scalar_features=10,
        freeze_encoders=True,
        use_text=True,
        use_vision=False  # Skip vision for speed
    )

    params = model.count_parameters()
    print(f"  ✓ Model initialized: {params['total']:,} total params")

    # Test forward pass (text-only)
    batch_size = 2
    dummy_text = {
        'input_ids': torch.randint(0, 30522, (batch_size, 128)),
        'attention_mask': torch.ones(batch_size, 128)
    }
    dummy_scalars = torch.randn(batch_size, 10)

    cls_logits, reg_output = model(
        text_input=dummy_text,
        image_input=None,
        scalar_features=dummy_scalars
    )

    assert cls_logits.shape == (batch_size, 2), f"Wrong classification shape: {cls_logits.shape}"
    assert reg_output.shape == (batch_size, 1), f"Wrong regression shape: {reg_output.shape}"
    print(f"  ✓ Forward pass: cls={cls_logits.shape}, reg={reg_output.shape}")


# ============================================================================
# TEST 9: TRAINING UTILITIES
# ============================================================================

@test_section("Training Utilities")
def test_training_utils():
    """Test training utility functions"""
    import torch
    from training.utils import (
        set_seed, get_device, get_optimizer,
        EarlyStopping, AverageMeter
    )

    print("Testing utility functions...")

    # Test seed setting
    set_seed(42)
    print("  ✓ Random seed set")

    # Test device detection
    device = get_device("cuda")
    print(f"  ✓ Device: {device}")

    # Test optimizer creation
    model = torch.nn.Linear(10, 2)
    optimizer = get_optimizer(model, "adamw", learning_rate=1e-3)
    print(f"  ✓ Optimizer: {optimizer.__class__.__name__}")

    # Test early stopping
    early_stop = EarlyStopping(patience=3, mode="max")
    scores = [0.5, 0.6, 0.65, 0.64, 0.63, 0.62]
    stopped = False
    for score in scores:
        if early_stop(score):
            stopped = True
            break
    assert stopped, "Early stopping didn't trigger"
    print(f"  ✓ Early stopping triggered after {early_stop.counter} epochs")

    # Test average meter
    meter = AverageMeter()
    for val in [1.0, 2.0, 3.0]:
        meter.update(val)
    assert abs(meter.avg - 2.0) < 1e-6, "Average calculation wrong"
    print(f"  ✓ AverageMeter: avg={meter.avg:.2f}")


# ============================================================================
# TEST 10: CONFIGURATION
# ============================================================================

@test_section("Configuration Loading")
def test_configuration():
    """Test configuration loading"""
    from training.utils import load_config

    print("Testing configuration loading...")
    config = load_config('src/configs/training_config.yaml')

    assert 'model' in config, "Missing model config"
    assert 'training' in config, "Missing training config"
    assert 'data' in config, "Missing data config"

    print(f"  ✓ Config loaded: {list(config.keys())}")
    print(f"  ✓ Model: {config['model']['name']}")
    print(f"  ✓ Epochs: {config['training']['epochs']}")
    print(f"  ✓ Batch size: {config['data']['batch_size']}")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print()
    print("Starting module validation...")
    print()

    # Run all tests
    test_imports()
    test_preprocessing()
    test_feature_engineering()
    test_pytorch_dataset()
    test_baseline_model()
    test_text_encoder()
    test_vision_encoder()
    test_fusion_model()
    test_training_utils()
    test_configuration()

    # Print summary
    print()
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print()

    if tests_failed > 0:
        print("❌ FAILED TESTS:")
        for name, error in errors:
            print(f"  - {name}")
            print(f"    Error: {error[:100]}")
        print()
        print("⚠️ Please fix the errors above before running in Colab!")
        sys.exit(1)
    else:
        print("✅ ALL TESTS PASSED!")
        print()
        print("Your code is ready to run in Google Colab! 🚀")
        print()
        print("Next steps:")
        print("1. Upload your code to Google Drive or GitHub")
        print("2. Open notebooks/phase1_training_colab.ipynb in Colab")
        print("3. Select A100 GPU runtime")
        print("4. Run all cells")
        print()
        sys.exit(0)
