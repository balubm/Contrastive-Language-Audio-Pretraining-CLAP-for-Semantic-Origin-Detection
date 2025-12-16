# CLAP-based Music Origin Detection: AI vs Human

This project leverages **Contrastive Language–Audio Pretraining (CLAP)** to distinguish AI-generated music from human-composed music using cross-modal audio-text embeddings. By comparing audio tracks to descriptive text labels, we extract interpretable similarity scores and use them as features for machine learning classification.

LinkedIn Article: https://www.linkedin.com/pulse/contrastive-languageaudio-pretraining-clap-semantic-origin-balu-pnztc/

---

## Project Overview

As AI music generation becomes increasingly sophisticated, differentiating AI-generated tracks from human compositions is essential for copyright integrity and artistic authenticity. This project:

- Extracts **CLAP audio embeddings** from both AI-generated (Suno) and human-composed (FMA) music.
- Computes **cosine similarity scores** between audio embeddings and textual descriptors like:
  - `"ai-generated music"` vs `"human-performed music"`
  - `"digital"` vs `"natural, live"`
  - `"music with audio glitches"` vs `"music with smooth production"`
  - `"synthetic music track"` vs `"music created by a musician"`
- Uses these similarity scores to train a **Support Vector Machine (SVM)** classifier to predict the source of music tracks.
- Provides **scatter plots and box plots** for visualization of similarity distributions and class separation.

---

## Dataset

- **AI-generated music (Suno)**: 21,400 tracks from a public Kaggle dataset.
- **Human music (FMA)**: 25,000 tracks from the Free Music Archive (medium subset), using 30-second clips.

> All Suno tracks were trimmed to their last 30 seconds to ensure comparable instrumentation across the dataset.

---

## Features & Classification

1. **Audio embeddings**: 512-dimensional vectors from CLAP.
2. **Text embeddings**: Embeddings of descriptive labels.
3. **Similarity scores**: Cosine similarity between audio and text embeddings, forming the feature set for classification.
4. **Machine learning model**: SVM with RBF kernel, tuned via grid search:
   - `C = 100`
   - `γ = 0.01`

**Performance**:
- Overall accuracy: **~93%**
- Confusion matrix (test set):

|                | Predicted Suno | Predicted Human |
|----------------|----------------|----------------|
| **Actual Suno** | 4,010          | 273            |
| **Actual Human** | 402           | 4,595          |

---

## Visualization

- Scatter plots of similarity scores for each label pair.
- Box plots showing the **highest similarity score per track**, separated by Suno vs Human sources.
- Internal annotations indicate the percentage of tracks predicted for each label.

---

## Usage

1. Install dependencies:

```bash
pip install torch torchaudio pandas seaborn matplotlib scikit-learn tqdm msclap
