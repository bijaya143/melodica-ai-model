from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from scipy.sparse import load_npz
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# pip install fastapi uvicorn // Packages
# uvicorn recommendation:app --reload // Run Command

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize Spotify client
CLIENT_ID = "70a9fb89662f4dac8d07321b259eaad7"
CLIENT_SECRET = "4d6710460d764fbbb8d8753dc094d131"
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load the data and similarity matrix
def load_data():
    with open('df.pkl', 'rb') as f:
        df = pickle.load(f)
    similarity = load_npz('similarity_sparse.npz').tocsr()
    return df, similarity

# Define a Pydantic model for search query
class SearchQuery(BaseModel):
    query: str

# Define a Pydantic model for song selection
class SongSelection(BaseModel):
    selected_song: str

# Function to get recommendations
def get_recommendations(selected_song, df, similarity, top_n=10):
    try:
        song_idx = df[df['track_name'] == selected_song].index[0]
    except IndexError:
        return []

    similarity_scores = similarity[song_idx].toarray()[0]
    sorted_indices = np.argsort(similarity_scores)[::-1]

    recommended_songs = []
    seen = set()
    total_recommendations = 0

    # Add unique songs first
    for i in range(1, len(sorted_indices)):
        if total_recommendations >= top_n:
            break
        idx = sorted_indices[i]
        song = df.iloc[idx]
        song_key = (song['track_name'], song['artists'])

        if song_key not in seen:
            seen.add(song_key)
            song_info = {
                "title": song['track_name'],
                "imageUrl": get_song_album_cover_url(song['track_name'], song['artists']),
                "artist": song['artists'],
                "similarity_score": similarity_scores[idx]
            }
            recommended_songs.append(song_info)
            total_recommendations += 1

    # If there are not enough unique songs, add the closest songs until the quota is filled
    if total_recommendations < top_n:
        for i in range(1, len(sorted_indices)):
            if total_recommendations >= top_n:
                break
            idx = sorted_indices[i]
            song = df.iloc[idx]
            song_key = (song['track_name'], song['artists'])

            # Re-add songs even if they are repeated
            song_info = {
                "title": song['track_name'],
                "imageUrl": get_song_album_cover_url(song['track_name'], song['artists']),
                "artist": song['artists'],
                "similarity_score": similarity_scores[idx]
            }
            recommended_songs.append(song_info)
            total_recommendations += 1

    # Fill the remaining spots with a fallback image if still not enough recommendations
    while len(recommended_songs) < top_n:
        recommended_songs.append({
            "title": "No more recommendations",
            "imageUrl": "https://i.postimg.cc/0QNxYz4V/social.png",
            "artist":"",
            "similarity_score": 0
        })

    return recommended_songs

# Function to get album cover URL
def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    try:
        results = sp.search(q=search_query, type="track")
        if results and results["tracks"]["items"]:
            track = results["tracks"]["items"][0]
            album_cover_url = track["album"]["images"][0]["url"]
            return album_cover_url
        else:
            return "https://i.postimg.cc/0QNxYz4V/social.png"  # Fallback image
    except Exception as e:
        return "https://i.postimg.cc/0QNxYz4V/social.png"  # Fallback image

# API endpoint for getting song options based on search query
@app.post("/get_options")
async def get_options(search_query: SearchQuery):
    df, _ = load_data()
    filtered_songs = df[df['track_name'].str.contains(search_query.query, case=False, na=False)]['track_name'].values[:10]
    
    if filtered_songs.size > 0:
        return {"songs": filtered_songs.tolist()}
    else:
        raise HTTPException(status_code=404, detail="No matching songs found")

# API endpoint for getting recommendations based on the selected song
@app.post("/get_recommendations")
async def get_song_recommendations(song_selection: SongSelection):
    df, similarity = load_data()
    recommended_songs = get_recommendations(song_selection.selected_song, df, similarity)
    
    if recommended_songs:
        return {"recommended_songs": recommended_songs}
    else:
        raise HTTPException(status_code=404, detail="No recommendations available")