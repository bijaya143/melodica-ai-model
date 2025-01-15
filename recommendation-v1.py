import pickle
import pandas as pd
import numpy as np
import streamlit as st
from scipy.sparse import load_npz

# Load data and similarity matrix with caching
@st.cache_data
def load_data():
    with open('df.pkl', 'rb') as f:
        df = pickle.load(f)
    similarity = load_npz('similarity_sparse.npz').tocsr()
    return df, similarity

@st.cache_data
def get_filtered_songs(query, df, limit=20):
    """Filter songs based on a search query."""
    if not query:
        return df['track_name'].head(limit).values
    return df[df['track_name'].str.contains(query, case=False, na=False)]['track_name'].head(limit).values

# Get recommendations for a selected song
def get_recommendations(selected_song, df, similarity, top_n=5):
    try:
        song_idx = df[df['track_name'] == selected_song].index[0]
    except IndexError:
        return [], []
    
    similarity_scores = similarity[song_idx].toarray()[0]
    sorted_indices = np.argsort(similarity_scores)[::-1]

    recommended_songs = []
    recommended_posters = []
    for i in range(1, min(top_n + 1, len(sorted_indices))):
        idx = sorted_indices[i]
        song = df.iloc[idx]
        recommended_songs.append(song['track_name'])
        recommended_posters.append(get_song_album_cover_url(song['track_name'], song['artists']))

    while len(recommended_songs) < top_n:
        recommended_songs.append("No more recommendations")
        recommended_posters.append("https://i.postimg.cc/0QNxYz4V/social.png")

    return recommended_songs, recommended_posters

def get_song_album_cover_url(song_name, artist_name):
    """Mock function for album cover URLs."""
    return "https://i.postimg.cc/0QNxYz4V/social.png"

# Streamlit UI
st.header('Music Recommender System')

# Load dataset and similarity matrix
df, similarity = load_data()
st.write(f"Loaded dataset with {df.shape[0]} songs.")

# Search functionality
query = st.text_input("Search for a song", "")
filtered_music_list = get_filtered_songs(query, df)

# Song selection
if len(filtered_music_list) > 0:
    selected_song = st.selectbox("Select a song to get recommendations", filtered_music_list)
else:
    selected_song = None
    st.warning("No songs match your search query.")

# Recommendations display
if st.button("Show Recommendations") and selected_song:
    with st.spinner("Fetching recommendations..."):
        recommended_songs, recommended_posters = get_recommendations(selected_song, df, similarity)

    if recommended_songs:
        st.subheader(f"Top 5 recommendations for '{selected_song}':")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                st.text(recommended_songs[i])
                st.image(recommended_posters[i])
    else:
        st.error("No recommendations available.")
