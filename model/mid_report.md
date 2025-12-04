# Multimodal Analysis of Short-Form Video Algorithms

```
Cheney Yoon
1007651177, cheney.yoon@mail.utoronto.ca
colab link | github link
```
## Brief Project Description

```
Short-form video platforms (YouTube Shorts, TikTok) drive massive cultural influence, yet their
ranking algorithms remain opaque. This project reverse-engineers viral video patterns using
multimodal deep learning to predict which videos will trend.
```
```
Why Deep Learning? Virality emerges from complex, non-linear interactions across text semantics,
engagement dynamics, and temporal patterns. Traditional feature engineering cannot capture these
high-dimensional relationships. Deep learning enables: (1) automatic feature extraction via pretrained
encoders (BERT for text) ( 1 ), (2) multimodal fusion to learn cross-modal interactions, and (3) joint
optimization for classification (viral/non-viral) and regression (engagement velocity).
```
## System Architecture:

```
Figure 1: System architecture overview showing multimodal fusion of BERT ( 1 ) text embeddings
and numeric features
```
## Data Processing

```
Source: YouTube Shorts & TikTok Trends 2025 (50k videos, CC0 license) (2).
```
```
Processing pipeline: (1) Language filtering (English only), (2) Null removal (critical fields: title,
views, likes), (3) Feature engineering (engagement_velocity =likes+commentsviews+1+shares), (4) Cate-
gorical encoding (day_of_week via LabelEncoder), (5) Normalization (StandardScaler for features,
MinMaxScaler for velocity→ [0,1]), (6) Stratified 70/15/15 train/val/test split.
```
```
0.1 Dataset statistics
```
```
Table 1: Dataset split statistics (20% viral, 80% non-viral)
```
```
Split Samples Viral (20%) Non-Viral (80%)
Train 6,679 1,336 5,
Val 1,432 286 1,
Test 1,431 286 1,
```
18 key features: views, likes, comments, shares, saves, engagement_rate, completion_rate, like_rate,
comment_ratio, share_rate, save_rate, upload_hour, day_of_week, is_weekend, duration_sec, ti-
tle_length, has_emoji, creator_avg_views.


Sample training example: Title: "POV: When your dog hears you open the snack drawer", views:
125,000, likes: 8,500, engagement_rate: 0.82 (normalized), duration_sec: 12, has_emoji: 1, up-
load_hour: 18. Label: Viral (1), Velocity: 0.85.
Challenges: (1) Class imbalance (80/20)→addressed with class weights, (2) Feature scale mismatch
(views: 106 , likes: 103 )→solved with StandardScaler, (3) Regression loss explosion (initial MSE:
511M)→ rebalanced loss weights (0.95 classification, 0.05 regression).
Future testing: New temporal split (post-training time period) and cross-platform evaluation (TikTok-
only) for generalization assessment.

## Baseline model

```
Architecture: Logistic Regression with TF-IDF features (5000 unigrams, balanced class weights).
Uses video titles only.
```
```
Figure 2: Baseline logistic regression architecture using TF-IDF text features only
```
```
Results:
```
```
Table 2: Baseline model performance
```
```
Metric Value Target
Test AUROC 0.488 ≥ 0.
Accuracy 0.561 –
Precision / Recall 0.23 / 0.69 –
```
```
Analysis: Baseline fails to reach target (0.488<0.65), performing barely better than random.
Text-only features lack semantic depth and miss critical engagement signals, justifying the need for
multimodal deep learning.
```
## Primary model

```
Architecture: Multimodal fusion network combining BERT text encoder with numeric features.
```
```
Figure 3: Detailed primary model architecture with layer dimensions and activation functions
```
```
Complexity: 110.5M total parameters (109.5M trainable). BERT unfrozen for task-specific fine-
tuning.
```
```
Training: AdamW (lr=2e-5, warmup 500 steps), batch size 32, 15 epochs, mixed precision (FP16),
gradient clipping (max_norm=1.0). Loss: 0. 95 × CrossEntropy + 0. 05 × M SEwith class weights
[0.625, 2.5].
```

Figure 4: Training progress over 7 epochs. Left: Validation AUROC reaches 0.845 by epoch 3,
exceeding target (0.75). Right: Training and validation loss curves showing convergence without
overfitting. Early stopping triggered at epoch 6.

Results:

```
Table 3: Primary model performance vs. baseline
```
```
Metric Multimodal Baseline Target Status
AUROC 0.855 0.488≥0.75✓+14%
Accuracy 0.802 0.561 – –
Precision 0.76 0.23 – –
F1 Score 0.72 0.34 – –
Velocity MAE 0.031 – ≤0.30✓ 10 ×
Velocity R^2 0.84 – – –
```
```
Figure 5: Confusion matrix for primary model on test set (n=1,431)
```
Qualitative analysis

Success: Correctly identifies viral videos with strong engagement + optimal timing.

False positives: Viral-like titles posted at suboptimal hours (3 AM uploads).

False negatives: Older viral content with decayed velocity metrics.

Feature ablation: Removing engagement rates→AUROC drops to 0.75 (−0.105). Removing text
(BERT)→ AUROC drops to 0.78 (−0.075). Engagement rates contribute most signal.

Implementation challenges: (1) Feature normalization crisis (all predictions viral without Standard-
Scaler), (2) Regression loss dominance (solved by target normalization + weight rebalancing), (3)
PyTorch 2.6 compatibility (torch.load() weights_only issue).

Progress:✓Data collected (9,542 videos),✓Baseline complete (AUROC 0.488),✓Primary model
exceeds target (AUROC 0.855 > 0.75),✓Training infrastructure operational (MLflow).
Remaining work: Hyperparameter tuning, ensemble methods, SHAP interpretability, error analysis,
cross-platform evaluation. Project on track with 2-3 weeks remaining for refinement.


## References

[1]J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional
transformers for language understanding,” in Proceedings of the 2019 Conference of the North
American Chapter of the Association for Computational Linguistics: Human Language Tech-
nologies (NAACL-HLT). Minneapolis, Minnesota: Association for Computational Linguistics,
2019, pp. 4171–4186, [Online]. Available: https://aclanthology.org/N19-1423/.

[2]T. Masryo, “Youtube shorts and tiktok trends 2025 dataset,” Hugging Face, 2025, [Online].
Available: https://huggingface.co/datasets/tarekmasryo/YouTube-Shorts-TikTok-Trends-2025.


