import os
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tqdm import tqdm


def extract_features(image_directory, output_path):
    # Load VGG16 model without the top layer
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    print("Feature extraction model loaded.")

    features = {}
    # Extract features from each image
    for img_name in tqdm(os.listdir(image_directory)):
        img_path = os.path.join(image_directory, img_name)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)

        # Use image ID as the key
        image_id = img_name.split('.')[0]
        features[image_id] = feature

    # Save features to a pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)

    print(f"Features extracted and saved to {output_path}")
    return features


def load_features(features_path):
    """Load image features from pickle file"""
    with open(features_path, 'rb') as f:
        return pickle.load(f)