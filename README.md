# Miles - Viral Shorts Analysis System

A multimodal deep learning system for predicting viral potential of short-form videos and providing actionable content recommendations to creators.

## Overview

This project reverse-engineers YouTube Shorts and TikTok algorithms through data-driven analysis, using deep learning to predict which videos will trend and providing creators with actionable insights.

**Key Features:**
- Multimodal ML model (BERT + ResNet-50) for virality prediction
- Real-time video scraping and analysis
- SHAP-based interpretability for understanding success factors
- LLM-powered content recommendations
- Web-based analytics dashboard

## Project Status

🚧 **Currently in Phase 1: Foundation & ML Core (Weeks 1-4)**

- ✅ Project structure and environment setup
- 🔄 Data pipeline implementation (in progress)
- ⏳ Model architecture development
- ⏳ Training infrastructure
- ⏳ Model training and evaluation

See [implementation_plan.md](implementation_plan.md) for the complete roadmap.

## Project Structure

```
Miles/
├── src/
│   ├── data/              # Data loading, preprocessing, feature engineering
│   ├── models/            # Model architectures (baseline, BERT, ResNet, fusion)
│   ├── training/          # Training loops, evaluation metrics, utilities
│   └── configs/           # Configuration files (YAML)
├── notebooks/             # Jupyter/Colab notebooks for experimentation
├── tests/                 # Unit tests
├── data/                  # Data storage (gitignored)
│   ├── raw/              # Original downloaded data
│   ├── processed/        # Cleaned and preprocessed data
│   └── splits/           # Train/val/test splits
├── experiments/           # MLflow experiment tracking (gitignored)
├── docs/                  # Additional documentation
└── [documentation files]
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended) or Google Colab Pro+
- Git and Git LFS (for large files)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Miles
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up HuggingFace authentication (for dataset access):
```bash
# Create .env file with your HuggingFace token
echo "HF_TOKEN=your_token_here" > .env
```

## Phase 1: ML Core Development

### Current Tasks (Week 1-2)

**Data Pipeline:**
- Download YouTube Shorts dataset from HuggingFace (~50k videos)
- Implement text preprocessing (BERT tokenization)
- Extract visual features from thumbnails
- Engineer engagement metrics

**Model Architecture:**
- Baseline logistic regression (target: AUROC ≥ 0.65)
- BERT text encoder (768-dim embeddings)
- ResNet-50 vision encoder (2048-dim embeddings)
- Multimodal fusion MLP

**Training Infrastructure:**
- MLflow experiment tracking
- Mixed precision training
- Checkpointing and early stopping
- Comprehensive evaluation metrics

### Success Criteria

Phase 1 will be considered complete when:
- ✅ Baseline model achieves AUROC ≥ 0.65
- ✅ Multimodal model achieves AUROC ≥ 0.75
- ✅ Velocity prediction MAE < 0.3
- ✅ Model inference < 100ms per video on GPU
- ✅ All unit tests pass with >80% coverage

## Usage

### Running Data Preprocessing
```python
from src.data.download import download_dataset
from src.data.preprocessing import preprocess_data

# Download dataset
dataset = download_dataset()

# Preprocess
preprocessed = preprocess_data(dataset)
```

### Training Models
```bash
# Train baseline model
python -m src.training.train --config configs/baseline_config.yaml

# Train multimodal model
python -m src.training.train --config configs/training_config.yaml
```

### Running Tests
```bash
pytest tests/ -v --cov=src
```

## Documentation

- [Project Proposal](APS360_project_proposal.md) - Academic project proposal
- [Overall Architecture](overall_architecture.md) - System architecture and design
- [Implementation Plan](implementation_plan.md) - Detailed 16-week roadmap

## Technology Stack

**ML & Deep Learning:**
- PyTorch 2.0+
- Hugging Face Transformers (BERT)
- TorchVision (ResNet-50)
- Scikit-learn

**Data & Infrastructure:**
- HuggingFace Datasets
- Pandas, NumPy
- MLflow (experiment tracking)
- DVC (data versioning)

**Development:**
- Google Colab Pro+ (training)
- pytest (testing)
- Git & GitHub (version control)

## License

This project is for educational purposes as part of APS360 (Applied Fundamentals of Deep Learning) at the University of Toronto.

## Contact

Cheney Yoon - cheney.yoon@mail.utoronto.ca

Project Link: [GitHub Repository URL]
