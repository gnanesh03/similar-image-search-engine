import numpy as np
from numpy.linalg import norm
import pickle
# from tqdm import tqdm, tqdm_notebook
#from tqdm import tqdm

import os
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


def extract_features(img_path, model):
    input_shape = (224, 224, 3)
    img = image.load_img(img_path, target_size=(
        input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features


extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
def get_file_list(root_dir):
    file_list = []
    counter = 1
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
                file_list.append(os.path.join(root, filename))
                counter += 1
    return file_list

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("loading the model")

    model = ResNet50(weights='imagenet', include_top=False,
                     input_shape=(224, 224, 3))

    root_dir="./images";
    filenames = sorted(get_file_list(root_dir))

    feature_list = []
    for i in range(5):
        feature_list.append(extract_features(filenames[i], model))

    # saving the compuatated feature float values
    pickle.dump(feature_list, open('test_data/features-caltech101-resnet.pickle', 'wb'))
    pickle.dump(filenames, open('test_data/filenames-caltech101.pickle', 'wb'))
