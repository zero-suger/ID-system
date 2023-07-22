import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import numpy as np

def initialize_firebase():
    cred = credentials.Certificate("Firebase-admin-conifs.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://id-verify-3bc2f-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })

def save_user_data(ref, user_data):
    existing_users = ref.get()

    if existing_users is not None:
        for user_id, data in existing_users.items():
            if data['name'] == user_data['name']:
                return  # User already exists, skip saving

    ref.push().set(user_data)

def get_users_data(ref):
    return ref.get()
