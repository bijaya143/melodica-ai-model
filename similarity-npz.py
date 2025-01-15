import pickle
from scipy.sparse import save_npz, csr_matrix

# Load the similarity matrix from the pickle file
with open('similarity_sparse.pkl', 'rb') as f:
    similarity = pickle.load(f)

# Convert the similarity matrix to CSR format
if not isinstance(similarity, csr_matrix):
    similarity = csr_matrix(similarity)

# Save the sparse similarity matrix as .npz
save_npz('similarity_sparse.npz', similarity)

print("Sparse similarity matrix saved as 'similarity_sparse.npz'")
