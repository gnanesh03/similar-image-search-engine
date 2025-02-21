import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from pinecone import Pinecone
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index_name = "similar-image-search"
index = pinecone_client.Index(pinecone_index_name)

# Load the ResNet50 model for feature extraction
# model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
# model = tf.keras.Sequential([
#     model,
#     tf.keras.layers.GlobalAveragePooling2D()
# ])


# Function to extract features from an image
# def extract_features(image_path):
#     img = image.load_img(image_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     expanded_img_array = np.expand_dims(img_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img_array)
#
#     features = model.predict(preprocessed_img).flatten()
#     normalized_features = features / norm(features)
#
#     return normalized_features


# Function to upload images in ./test_images to Pinecone
# def upload_all_images(directory="./test_images"):
#     if not os.path.exists(directory):
#         print(f"Directory '{directory}' does not exist.")
#         return
#
#     image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#
#     for image_file in image_files:
#         image_path = os.path.join(directory, image_file)
#         features = extract_features(image_path)
#
#         # Use filename (without extension) as the ID
#         image_id = os.path.splitext(image_file)[0]
#
#         # Store in Pinecone
#         index.upsert(vectors=[(image_id, features.tolist())])
#         print(f"Uploaded: {image_file} -> ID: {image_id}")


# Function to query similar images from Pinecone
# def query_similar_images(image_path, top_k=5):
#     features = extract_features(image_path)
#     results = index.query(vector=[features.tolist()], top_k=top_k)
#
#     similar_images = [match['id'] for match in results['matches']]
#     print(f"Query Image: {image_path}")
#     print(f"Similar Images: {similar_images}")
#     return similar_images


# Run script
if __name__ == "__main__":
    # Delete all vectors from the index
    index.delete(delete_all=True)

    # Upload all images
    # print("Uploading all images to Pinecone...")
    # upload_all_images()

    # Query using a test image
    # test_image_path = "C:/Users/KIYOTAKA/Desktop/image_114.jpg"  # Change this to an actual image in test_images
    # print("\nQuerying similar images...")
    # print(query_similar_images(test_image_path))
