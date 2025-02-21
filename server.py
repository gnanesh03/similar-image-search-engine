import os
import io
import numpy as np

import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from pinecone import Pinecone
from fastapi import FastAPI, UploadFile, File,Form
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()




pinecone_client=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index_name = "similar-image-search"
index = pinecone_client.Index(pinecone_index_name)


#stats = index.describe_index_stats()
#print(f"Total Records: {stats['total_vector_count']}")




# Load the ResNet50 model (exclude the top layer for feature extraction) with global average pooling
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Sequential([
    model,
    tf.keras.layers.GlobalAveragePooling2D()  # This reduces the shape to (2048,)
])


# Function to extract features from an image
def extract_features(img_data):
    # Load the image using Keras' image loading functions
    img = image.load_img(io.BytesIO(img_data), target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    # Predict features using ResNet50
    features = model.predict(preprocessed_img).flatten()

    # Normalize the features
    normalized_features = features / norm(features)

    return normalized_features


def store_features_in_pinecone(features, post_id, url):
    # Your existing Pinecone storage logic
    vector = {
        "id": url,  # Store url as the vector ID
        "values": features,  # Store extracted feature vector
        "metadata": {"post_id": post_id}  # Store image URL as metadata
    }

    # Insert the vector into Pinecone
    index.upsert([vector])


# Function to perform similarity search in Pinecone
def search_similar_images(query_features, top_k=10):

    vector = query_features.tolist();#because numpy needs to be changed to list to pass it
    results = index.query(vector=vector, top_k=top_k,include_metadata=True)
    return results['matches']


@app.post("/search-image")
async def search_similar_image(image: UploadFile = File(...)):
    # Read the image data
    image_data = await image.read()

    # Extract features from the image
    query_features = extract_features(image_data)
    #print(query_features)
    #return{"features:"}

    # Search for similar images in Pinecone
    similar_images = search_similar_images(query_features)

    # Return the results (urls and ids of similar images)

    # for e in similar_images:
    #     print(e)

    results = [{"post_id": match["metadata"]["post_id"], "url": match["id"]} for match in similar_images]
    #print(results)
    return {"similar_images": results}

@app.post("/upload-image")
async def upload_image(
    image: UploadFile = File(...),
    post_id: str = Form(...),
    url: str = Form(...)  # Receiving the original image URL
):
    # Read the image data
    image_data = await image.read()

    # Extract features from the image
    features = extract_features(image_data)

    # Store the features in Pinecone with post_id and url
    store_features_in_pinecone(features, post_id, url)

    return {"message": f"Image {post_id} uploaded and features stored in Pinecone"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
