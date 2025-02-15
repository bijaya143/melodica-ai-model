import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix  # Sparse matrix format for efficiency

# Function to reduce memory usage (optional but recommended)
def reduce_memory_usage(df):
    print("Reducing memory usage...")
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:  # If not a string/object column
            if col_type in ['int64', 'int']:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif col_type in ['float64', 'float']:
                df[col] = pd.to_numeric(df[col], downcast='float')
        else:
            df[col] = df[col].astype('category')  # Convert objects to categories
    print("Memory usage reduced!")
    return df

# Step 1: Load the preprocessed dataset from pickle
pickle_file_path = "df.pkl"  # Replace with your pickle file path
print("Loading dataset from pickle...")
with open(pickle_file_path, "rb") as f:
    data = pickle.load(f)
print(f"Dataset loaded with shape: {data.shape}")

# Step 2: Select relevant features for similarity calculation
features = ['valence', 'loudness', 'key', 'speechiness', 'acousticness']  
print(f"Using features for similarity: {features}")
if not all(feature in data.columns for feature in features):
    raise ValueError("Some selected features are not present in the dataset!")

# Normalize the features (scaling values to 0–1 for consistent comparison)
print("Normalizing feature values...")
normalized_data = data[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Step 3: Initialize a sparse similarity matrix (to save memory)
similarity_matrix = lil_matrix((data.shape[0], data.shape[0]))  # Use LIL format for sparse matrix

# Step 4: Define chunk size for calculating similarity in parts
chunk_size = 1000  # Modify this based on your available memory
print(f"Using chunk size: {chunk_size}")

# Step 5: Calculate cosine similarity matrix in chunks
print("Calculating cosine similarity matrix in chunks...")
for start_idx in range(0, data.shape[0], chunk_size):
    end_idx = min(start_idx + chunk_size, data.shape[0])
    chunk = normalized_data.iloc[start_idx:end_idx]
    
    # Calculate similarity for the current chunk
    chunk_similarity = cosine_similarity(chunk)
    
    # Update the overall similarity matrix (only store in sparse format)
    similarity_matrix[start_idx:end_idx, start_idx:end_idx] = chunk_similarity

    print(f"Processed chunk {start_idx} to {end_idx}")

# Step 6: Save the sparse similarity matrix to pickle
similarity_pickle_path = "similarity_sparse.pkl"  # Desired output pickle file
print("Saving sparse similarity matrix to pickle...")
with open(similarity_pickle_path, "wb") as f:
    pickle.dump(similarity_matrix, f)
print(f"Sparse similarity matrix saved as: {similarity_pickle_path}")
