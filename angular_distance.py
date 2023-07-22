import numpy as np


# Load the embeddings from the .npy files
embedding1 = np.load('test01.npy')
embedding2 = np.load('test02.npy')

# Normalize the embeddings
embedding1 = embedding1 / np.linalg.norm(embedding1)
embedding2 = embedding2 / np.linalg.norm(embedding2)

# Compute the angular distance using the ArcFace loss formula
angular_distance = np.arccos(np.dot(embedding1, embedding2.T))

# Calculate the similarity score
similarity_score = 1 - angular_distance / np.pi

np.save('similarity_score.npy', similarity_score)
