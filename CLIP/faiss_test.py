import numpy as np
import faiss

# Create some sample vectors
vectors = np.random.rand(1000, 128).astype('float32')

# Create an index
index = faiss.IndexFlatL2(128)  # L2 distance for 128-dimensional vectors

# Add the vectors to the index
index.add(vectors)

# Search for the 5 nearest neighbors of a sample vector
query_vector = np.random.rand(1, 128).astype('float32')
distances, indices = index.search(query_vector, 5)

print(f"Nearest Neighbors: {indices[0]}")
print(f"Distances: {distances[0]}")
