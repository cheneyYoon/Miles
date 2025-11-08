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

🚧 **Currently done Phase 1: Foundation & ML Core**

- ✅ Project structure and environment setup
- ✅ Data pipeline implementation (in progress)
- ✅ Model architecture development
- ✅ Training infrastructure
- ✅ Model training and evaluation

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
├── experiments/           # MLflow experiment tracking (gitignored)
├── docs/                  # Additional documentation
└── [documentation files]
```

## License

This project is for educational purposes as part of APS360 (Applied Fundamentals of Deep Learning) at the University of Toronto.

## Contact

Cheney Yoon - cheney.yoon@mail.utoronto.ca

Project Link: [GitHub Repository URL]
