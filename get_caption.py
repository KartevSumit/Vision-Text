import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model

from model import predict_caption
from main import load_captions, clean_captions

BASE_DIR = 'D:/Program/archive'
WORKING_DIR = './Data'
MODEL_PATH = os.path.join(WORKING_DIR, 'best_model.h5')
CAPTIONS_FILE = os.path.join(BASE_DIR, 'captions.txt')
MAX_LENGTH = 34


def extract_feature_single(image_path):
    """Extract feature vector for a single image using VGG16."""
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)

    feature = model.predict(img, verbose=0)
    return feature


def load_tokenizer(captions_file):
    """Load and fit tokenizer on all training captions."""
    mapping = load_captions(captions_file)
    clean_captions(mapping)

    all_captions = []
    for captions in mapping.values():
        all_captions.extend(captions)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer


def generate_caption_for_image(image_path):
    """Given an image path, generate a caption."""
    model = load_model(MODEL_PATH)
    tokenizer = load_tokenizer(CAPTIONS_FILE)
    feature = extract_feature_single(image_path)

    caption = predict_caption(model, feature, tokenizer, MAX_LENGTH)
    return caption


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('image', help="Path to the input image")
    args = parser.parse_args()

    caption = generate_caption_for_image(args.image)
    print("Generated Caption:", caption)
