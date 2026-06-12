# Spotify Genre Segmentation & Music Recommendation System

An unsupervised machine learning project that clusters ~32,000 Spotify tracks by audio features and builds a **content-based song recommendation engine**. Uses K-Means clustering, PCA dimensionality reduction, and Euclidean distance similarity to recommend songs that sound like ones you already like.

---

## 🎵 What It Does

- Loads and preprocesses a 32,000-track Spotify dataset with 12 audio features
- Performs EDA on audio properties — danceability, energy, valence, tempo, and more
- Reduces dimensions with **PCA** for visualization and variance analysis
- Determines optimal clusters using the **Elbow Method + Silhouette Score**
- Clusters songs into 6 groups using **K-Means** (matching the 6 actual genres)
- Validates cluster quality by comparing ML clusters against real playlist genres
- Powers a **recommendation function** — input any song name, get 5 similar songs back

---

## 🎧 Demo

```python
recommend_songs('bad guy', df, X_scaled, num_recommendations=5)
```

```
   track_name              track_artist    playlist_genre  similarity_distance
0  Bury a Friend           Billie Eilish   pop             1.842
1  you should see me...    Billie Eilish   pop             2.103
2  wish you were gay       Billie Eilish   pop             2.341
3  bellyache               Billie Eilish   pop             2.519
4  when the party's over   Billie Eilish   pop             2.687
```

---

## 📊 Methodology

```
Raw Audio Features (12)
        │
        ▼
StandardScaler          ← Normalize feature scales
        │
        ▼
K-Means Clustering      ← Group songs by audio similarity (k=6)
        │
   ┌────┴────┐
   ▼         ▼
PCA (2D)   Euclidean   ← Visualization / Recommendation
Scatter    Distance
```

**Why k=6?** The dataset has 6 playlist genres. The Elbow Method and Silhouette Score both support this choice.

---

## 📈 Key Results

> Run the notebook to generate your own results.
- **Silhouette Score (k=6):** ~0.08–0.12 *(typical for music data — genre boundaries are inherently fuzzy)*
- **PCA 2-component variance:** ~35–42%
- **Cross-tab analysis:** Clusters show clear genre dominance despite overlap

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | scikit-learn (KMeans, PCA, StandardScaler) |
| Similarity | Euclidean Distance |
| Environment | Jupyter Notebook |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run
```bash
git clone https://github.com/akashcpatil111/spotify-genre-segmentation.git
cd spotify-genre-segmentation
jupyter notebook spotify_genre_segmentation.ipynb
```

> Make sure `spotify dataset.csv` is in the same directory as the notebook.

---

## 📁 Project Structure

```
spotify-genre-segmentation/
├── spotify_genre_segmentation.ipynb   ← Main notebook
├── spotify dataset.csv                ← Dataset (~32,000 Spotify tracks)
└── README.md
```

---

## 📋 Dataset

**Source:** [Kaggle — Spotify Dataset](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs)

**Audio Features Used:**
| Feature | Description |
|---|---|
| danceability | How suitable a track is for dancing (0–1) |
| energy | Perceptual intensity and activity (0–1) |
| loudness | Overall loudness in dB |
| speechiness | Presence of spoken words (0–1) |
| acousticness | Acoustic confidence score (0–1) |
| instrumentalness | Predicts no vocals (0–1) |
| liveness | Detects live audience presence (0–1) |
| valence | Musical positiveness (0–1) |
| tempo | Estimated beats per minute |
| duration_ms | Track length in milliseconds |

---

## 🔮 Potential Extensions

- Add DBSCAN or Hierarchical clustering for comparison
- Deploy recommendation function as a Flask/FastAPI REST endpoint
- Integrate with Spotify API for real-time recommendations
- Add collaborative filtering layer on top of content-based clustering

---

## 📄 License

MIT License
