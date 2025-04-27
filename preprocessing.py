import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model


def extract_image_features(base_dir, out_file):
    """Extract VGG16 features for images in base_dir/Images and save to pickle."""
    vgg = VGG16(weights='imagenet')
    extractor = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)
    features = {}
    img_folder = os.path.join(base_dir, 'Images')
    for img_name in tqdm(os.listdir(img_folder), desc='Extracting features'):
        img_path = os.path.join(img_folder, img_name)
        img = load_img(img_path, target_size=(224, 224))
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        feat = extractor.predict(arr, verbose=0)
        img_id = os.path.splitext(img_name)[0]
        features[img_id] = feat
    with open(out_file, 'wb') as f:
        pickle.dump(features, f)
    return features


def load_features(file_path):
    """Load pickled image features."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_captions(captions_file):
    """Read captions file and map image IDs to cleaned captions using startseq/endseq tokens."""
    mapping = {}
    with open(captions_file, 'r') as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, caption = line.split('.', 1)
            img_id = os.path.splitext(img_id)[0]
            words = [w for w in caption.lower().split() if len(w) > 1]
            cleaned = 'startseq ' + ' '.join(words) + ' endseq'
            mapping.setdefault(img_id, []).append(cleaned)
    return mapping


def create_tokenizer(captions_map):
    """Fit a tokenizer on all captions."""
    all_captions = [cap for caps in captions_map.values() for cap in caps]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer


def max_caption_length(captions_map):
    """Return the maximum caption length (in words)."""
    return max(len(cap.split()) for caps in captions_map.values() for cap in caps)