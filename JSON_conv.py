import numpy as np
import json

# Load the numpy array from the .npy file
embedding = np.load('image21.npy')

# Convert the numpy array to a JSON-serializable format
embedding_json = embedding.tolist()

# Save the JSON-serializable embedding to a file
with open('embedding.json', 'w') as file:
    json.dump(embedding_json, file)
