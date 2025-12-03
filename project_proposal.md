# Multimodal Analysis of Short-Form Video Algorithms

```
Cheney Yoon
1007651177, cheney.yoon@mail.utoronto.ca
https://tinyurl.com/3n9cnb6s
```
## Introduction

Short-form video surfaces such as YouTube Shorts and TikTok shape cultural trends and creator
livelihoods. Yet their proprietary ranking algorithms remain opaque. I propose a data-driven
reverse-engineering study that learns the patterns shared by "algorithm-boosted" videos and provides
actionable guidance to creators. Deep learning is appropriate because success signals are buried in
high-dimensional, multi-modal inputs (text, image, engagement traces) that show complex non-linear
interactions.

## Illustration

```
Figure 1: End-to-End Pipeline for Social Media Algorithm Analysis Tool
```
## Background & Related Work

1. Vallet et al. demonstrated that viral video prediction across platforms is feasible using cross-
    system features from Twitter to predict YouTube virality, achieving reasonable accuracy with
    traditional machine learning approaches but relying primarily on handcrafted signals (1).
2. Zhang et al. employed graph neural networks for viral rumor prediction on social media,
    demonstrating significant performance gains through multi-task learning approaches that
    jointly predict virality and user vulnerability (2).
3. Chen investigated TikTok’s self-improving algorithm using extensive user interaction analy-
    sis, revealing how Graph Neural Networks, Reinforcement Learning, and Temporal Convo-
    lutional Networks enable real-time content recommendation optimization (3).
4. Xie and Liu developed PrecWD, a deep learning framework for YouTube viewership predic-
    tion that combines structured metadata with unstructured video content, achieving superior
    performance while maintaining interpretability through attention mechanisms (4).


5. Ofli et al. proposed multimodal deep learning architectures combining CNN and text analysis
    for social media content analysis, demonstrating that joint representation learning from both
    visual and textual modalities significantly outperforms single-modality approaches (5).

This project extends these findings with a unified transformer-CNN architecture and public, cleaned
data.

## Data Processing

Data Sources

- Primary source: YouTube Shorts & TikTok Trends 2025 dataset, containing approximately
    50,000 videos with 32 cleaned features. Licensed under CC0.(6)
- Auxiliary source: YouTube-8M (4.9M YouTube video IDs with precomputed audio–visual
    embeddings and metadata) (7).

Data Cleaning Plan

- Deduplicate videos by their unique video ID.
- Exclude non-English titles for simplicity.
- Strip URLs and emojis from text fields; apply WordPiece tokenization, retaining the top
    50,000 vocabulary terms.
- Extract five dominant colors and entropy measures from video thumbnails; all thumbnails
    resized to 224×224 pixels.
- Standardize engagement metrics by normalizing counts relative to hours since upload, to
    remove age biases.
- Store the cleaned datasets in Apache Parquet format along with a comprehensive data
    dictionary on GitHub for reproducibility.

## Architecture

This project will employ a multimodal deep learning architecture with three main components.
The text metadata will be processed by a fine-tuned pre-trained BERT-base transformer to encode
semantic features. Visual features will be extracted from video thumbnails using a ResNet-50 CNN
pretrained on ImageNet. Scalar engagement features such as normalized views, likes, and shares
will be vectorized and concatenated with text and vision embeddings. A multilayer perceptron
with multiple fully connected layers will fuse these modalities and output a binary classification
predicting whether a video will trend or not, augmented by a regression head predicting a normalized
engagement velocity.

## Baseline Model

A traditional baseline model will be constructed using logistic regression with TF-IDF unigram
features extracted from video titles and hashtags. This transparent model sets a reference AUROC of
approximately 0.65 based on comparable prior work. Comparing deep learning performance against
this baseline will demonstrate the added value of sophisticated multimodal embedding approaches.

## Ethical Considerations

The datasets consist only of publicly available video metadata and thumbnails, excluding any person-
ally identifiable user information. The project will comply fully with platform terms of service and
ethical guidelines concerning data privacy. Thoughtful attention will be given to potential negative
consequences such as encouraging click-bait or homogeneity in content caused by algorithm gaming.
Bias analysis will be performed across demographics and content categories to ensure fairness.

### 2


## References

[1]D. Vallet, S. Berkovsky, S. Ardon, A. Mahanti, and M. A. Kafar, “Characterizing and predicting
viral-and-popular video content,” in Proceedings of the 24th ACM International Conference on
Information and Knowledge Management, 2015, pp. 1481–1490.

[2]X. Zhang, F. Wang, and T. Li, “Predicting viral rumors and vulnerable users with graph neural
networks,” arXiv preprint arXiv:2401.09724, 2024.

[3]X. Chen, “Investigation on the self-improving algorithm of tiktok extensive user interactions,” in
Proceedings of the International Conference on Knowledge Discovery and Information Retrieval,
2024, pp. 295–302.

[4]J. Xie and X. Liu, “Unbox the black-box: Predict and interpret youtube viewership using deep
learning,” Information Systems Research, vol. 32, no. 4, pp. 1215–1235, 2020.

[5]F. Ofli, F. Alam, and M. Imran, “Analysis of social media data using multimodal deep learning for
disaster response,” in Proceedings of the 17th International Conference on Information Systems
for Crisis Response and Management, 2020, pp. 1–12.

[6]T. Masryo, “Youtube shorts and tiktok trends 2025 dataset,” Hugging Face, 2025, [Online].
Available: https://huggingface.co/datasets/tarekmasryo/YouTube-Shorts-TikTok-Trends-2025.

[7]Google Research, “Youtube-8m large-scale video understanding dataset,” https://research.google.
com/youtube8m/, 2018, accessed: Sep. 26, 2025.

### 3


