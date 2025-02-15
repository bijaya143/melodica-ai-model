# Mood-Based Music Recommendation System

## 📌 Overview

The **Mood-Based Music Recommendation System** is a project that utilizes machine learning techniques to recommend music based on a user's mood. It leverages a **music dataset** and the **Spotify API** to analyze song attributes and generate personalized recommendations.

## 🎯 Features

- 🎵 **Mood-based song recommendations**
- 📊 **Analyzes song attributes** (e.g., loudness, speechiness, valence, acousticness, liveness)
- 🔍 **Spotify API integration** for real-time song fetching
- 🧠 **Machine Learning models** for mood classification
- 📂 **Uses a pre-existing dataset from Spotify** (available on Kaggle)
- 📈 **Content-based filtering for personalized recommendations**

## 🚀 Technologies Used

- **Python** (Pandas, NumPy, Scikit-learn)
- **FastAPI** (for API development)
- **Spotify API** (to fetch song details)
- **Jupyter Notebook** (for data analysis & model training)

## 📂 Project Structure

```
📁 mood-music-recommendation
│── 📂 dataset/                # Music dataset (CSV files)
│── 📂 outdated-models/                 # Machine learning models
│── 📂 frontend/               # Frontend React app
│── 📜 requirements.txt        # Python dependencies
│── 📜 README.md               # Project documentation
│── 📜 app.py                  # API entry point
│── 📜 model_training.ipynb    # Jupyter notebook for ML training
```

## 🎵 How It Works

1. **Data Preprocessing:** Extract relevant features from the music dataset.
2. **Mood Classification:** Train a machine learning model to classify songs into different moods.
3. **Spotify API Fetching:** Retrieve additional metadata for enhanced recommendations.
4. **Recommendation Engine:** Use content-based filtering to recommend songs similar to a given mood.

## 🛠 Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/mood-music-recommendation.git
   cd mood-music-recommendation
   ```

2. Run the server:
   ```bash
   python app.py
   ```

## 📊 Dataset

- The dataset used for training and recommendations is sourced from **Kaggle’s Spotify dataset**.
- It includes features such as **tempo, valence, energy, danceability, acousticness, and more**.

_🎵 Find the perfect music for your mood!_ 🎶
