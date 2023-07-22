import cv2
import numpy as np
from scipy.spatial.distance import cdist
from server_connect import initialize_firebase, save_user_data, get_users_data
from firebase_admin import db

new_image = cv2.imread("cropped_test01.jpg")

# Initialize Firebase
initialize_firebase()

ref = db.reference('users')


# Load the .npy files for multiple users
user1_embeddings = []
for i in range(1, 31):
    filename = f"./conv_npy/Aziz/{i}.npy"
    embedding = np.load(filename)
    user1_embeddings.append(embedding)

for i in range(100, 130):
    filename = f"./conv_npy/Aziz/{i}.npy"
    embedding = np.load(filename)
    user1_embeddings.append(embedding)


user2_embeddings = [
    np.load('./conv_npy/John/29562.npy'),
    np.load('./conv_npy/John/29563.npy'),
    np.load('./conv_npy/John/29564.npy')
]

# Convert the embeddings to lists
user1_embeddings_list = [embedding.tolist() for embedding in user1_embeddings]
user2_embeddings_list = [embedding.tolist() for embedding in user2_embeddings]

# User data
user_data1 = {
    'name': 'Azizbek',
    'embeddings': user1_embeddings_list
}

user_data2 = {
    'name': 'John',
    'embeddings': user2_embeddings_list
}

# Save the user data to Firebase
save_user_data(ref, user_data1)
save_user_data(ref, user_data2)

new_embedding = np.load('test01.npy')

users_data = get_users_data(ref)

embeddings_list = []
user_names = []

for user_id, user_data in users_data.items():
    embeddings_list.append([np.array(embedding) for embedding in user_data['embeddings']])
    user_names.append(user_data['name'])

similarity_scores = []

for embeddings, user_name in zip(embeddings_list, user_names):
    max_similarity_score = 0

    for embedding in embeddings:
        embedding = embedding.flatten()
        new_embedding = new_embedding.flatten()
        normalized_embedding = embedding / np.linalg.norm(embedding)
        normalized_new_embedding = new_embedding / np.linalg.norm(new_embedding)
        angular_distance = np.arccos(np.dot(normalized_embedding, normalized_new_embedding.T))
        similarity_score = (np.cos(angular_distance) + 1) / 2

        max_similarity_score = max(max_similarity_score, similarity_score)

    similarity_scores.append((user_name, max_similarity_score))

best_match = max(similarity_scores, key=lambda x: x[1])
matched_user_name, similarity_score = best_match
similarity_percentage = similarity_score * 100

if similarity_percentage < 70:
    text = "Input image is not in the database"
else:
    text = f"{similarity_percentage}"

# Put the text on top of the image
image_with_text = cv2.putText(new_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with the similarity score
cv2.imshow("Matching Result", image_with_text)
cv2.waitKey(0)
cv2.destroyAllWindows()