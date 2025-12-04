# Miles Project Status

## Project Overview

Miles is a multimodal deep learning system for predicting viral potential of short-form videos (YouTube Shorts/TikTok). The system combines BERT text encoders and ResNet-50 vision encoders to predict virality and provide actionable recommendations to content creators.

**Academic Context**: APS360 (Applied Fundamentals of Deep Learning) final project at University of Toronto

## What Currently Exists

### Core ML Infrastructure (Phase 1 - Complete)

**Data Pipeline** (~1,800 lines)
- Dataset loading and preprocessing with train/val/test splits
- Feature engineering for engagement metrics (velocity, rates, etc.)
- Dataset adapters for PyTorch DataLoader integration
- Data download utilities for video thumbnails and metadata
- Configured for parquet format with image directories

**Model Architecture** (~1,200 lines)
- Text encoder using BERT (bert-base-uncased) with frozen weights
- Vision encoder using ResNet-50 with frozen backbone
- Multimodal fusion model combining text, vision, and scalar features
- Dual-head output: binary classification (viral/not viral) + velocity regression
- Baseline model (Logistic Regression + TF-IDF) for comparison

**Training Infrastructure** (~1,500 lines)
- Complete training loop with validation and checkpointing
- MLflow experiment tracking integration
- Evaluation metrics: AUROC, accuracy, precision, recall, F1, MAE
- Early stopping and learning rate scheduling
- Mixed precision training support
- Model evaluation utilities

**Configuration & Environment**
- YAML-based training configuration (15 epochs, AdamW optimizer, etc.)
- Environment setup scripts (setup_local.sh, validate.sh)
- Dependencies specified (PyTorch, Transformers, MLflow, etc.)
- Git repository initialized

**Notebooks & Scripts**
- phase1_dry_run_local.ipynb: Local experimentation
- phase1_training_colab.ipynb: Colab training workflow
- generate_report_figures.py: Report visualization script

## What's Missing

### Immediate Priorities for Final Report

**Data Collection & Results** (CRITICAL)
- No actual data in the repository yet (data/ directories are empty)
- Need to collect/scrape YouTube Shorts or TikTok video data
- Need to run training on real data and generate results
- Must demonstrate model performance on validation and test sets

**Final Report Documentation** (Due Soon)
- 4-page LaTeX document required (64 points total)
- Must include: Introduction (2), Illustration (2), Related Work (2), Data Processing (4), Architecture (4), Baseline (4), Quantitative Results (4), Qualitative Results (4), New Data Evaluation (10), Discussion (8), Ethics (2), Difficulty/Quality (6), Structure/Grammar (8)
- Need actual training curves, confusion matrices, sample predictions
- Need discussion of model performance and limitations
- Need ethical considerations around recommendation algorithms

**Missing Report Artifacts**
- Training/validation loss curves and metrics plots
- Confusion matrix and classification examples
- Sample predictions showing successful and failed cases
- Data statistics and distribution visualizations
- Architecture diagram (hand-drawn or PowerPoint acceptable)

### Phase 2-4 Features (Post-Report)

**Phase 2: Interpretability**
- SHAP value analysis for understanding virality factors
- Feature importance visualization
- Model explanation interface

**Phase 3: Recommendations**
- LLM integration (GPT-4/Claude) for content suggestions
- Template generation for successful content patterns
- Actionable insights for creators

**Phase 4: Deployment**
- Web scraping system for real-time data
- API endpoints for predictions
- Web-based analytics dashboard
- Production-ready deployment infrastructure

### Technical Gaps

**Testing & Validation**
- Only basic module tests exist (tests/test_modules_quick.py)
- No integration tests for full pipeline
- No performance benchmarking on inference time

**Documentation**
- No docs/ directory with API documentation
- Missing docstrings in many modules
- No usage examples or tutorials

**Configuration**
- Missing experiment tracking in experiments/ (gitignored but empty)
- No DVC pipelines configured despite dvc in requirements
- No configs for different model variants (text-only, vision-only, etc.)

## Recommended Next Steps

1. **Collect Data** - Scrape 5,000-10,000 YouTube Shorts with metadata and thumbnails
2. **Run Training** - Execute full training pipeline and generate checkpoints
3. **Generate Results** - Create all figures and metrics for report
4. **Write Report** - Complete 4-page LaTeX document following rubric
5. **Evaluate on New Data** - Test model on fresh samples not used in training

## Technical Notes

**Model Specs**
- BERT: bert-base-uncased (110M params, frozen)
- ResNet-50: ImageNet pretrained (25M params, frozen)
- Fusion layers: 1024 → 256 → outputs (trainable)
- Total trainable params: ~2-3M

**Training Setup**
- Batch size: 32
- Learning rate: 2e-5 with warmup
- Early stopping: patience=3, monitor=AUROC
- Hardware: CUDA/MPS support with mixed precision

**Success Criteria**
- Baseline AUROC: 0.65
- Multimodal AUROC: 0.75
- Velocity MAE: 0.3
- Inference: <100ms

## Project Health

**Strengths**
- Clean, modular codebase with good separation of concerns
- Comprehensive ML infrastructure ready for training
- Well-configured hyperparameters and training settings
- Professional development setup (git, mlflow, testing framework)

**Risks**
- No real data collected yet (blocks all results)
- Final report deadline approaching with no trained models
- Ambitious scope (Phase 2-4) may not be achievable in timeframe
- Data scraping could face API rate limits or legal issues

**Overall Status**: Foundation is solid, but critical path requires immediate data collection and training execution to meet academic deliverables.
