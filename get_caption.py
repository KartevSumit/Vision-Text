import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from model import predict_caption
from main import load_captions, clean_captions

# Configure GPU memory management quietly
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception:
        pass

# Directory and file paths
BASE_DIR = 'D:/Program/archive'
WORKING_DIR = './Data'
MODEL_PATH = os.path.join(WORKING_DIR, 'best_model.h5')
CAPTIONS_FILE = os.path.join(BASE_DIR, 'captions.txt')
MAX_LENGTH = 34

# Global references to avoid reloading models and tokenizer
vgg_feature_model = None
caption_model = None
tokenizer = None


def extract_feature_single(image_path):
    """Extract feature vector for a single image using VGG16 on CPU."""
    global vgg_feature_model

    if vgg_feature_model is None:
        with tf.device('/CPU:0'):
            base_model = VGG16(weights='imagenet')
            vgg_feature_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape((1, *img.shape))
    img = preprocess_input(img)

    with tf.device('/CPU:0'):
        feature = vgg_feature_model.predict(img, verbose=0, batch_size=1)

    return feature


def load_tokenizer_once():
    """Load and fit tokenizer once on all captions."""
    global tokenizer

    if tokenizer is None:
        mapping = load_captions(CAPTIONS_FILE)
        clean_captions(mapping)
        all_captions = []
        for captions in mapping.values():
            all_captions.extend(captions)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_captions)

    return tokenizer


def clean_caption(caption):
    """Remove startseq and endseq tokens from caption and any file extension prefixes."""
    caption = caption.replace('startseq ', '').replace('endseq', '')

    file_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff',]
    for ext in file_extensions:
        flag = 0,
        for i in range(len(ext)):
            if ext[i] != caption[i]:
                break
            elif i == len(ext)-1 and ext[i] == caption[i]:
                flag = 1
        if flag == 1:
            caption = caption[len(ext):]

    caption = ' '.join(caption.split())
    caption = caption.strip()

    if caption:
        caption = caption[0].upper() + caption[1:]

    if caption and caption[-1] not in ['.', '!', '?']:
        caption += '.'

    return caption


def generate_caption_for_image(image_path):
    """Generate a caption for the given image using preloaded models."""
    global caption_model, tokenizer, vgg_feature_model

    # Load models and tokenizer once
    if caption_model is None:
        caption_model = load_model(MODEL_PATH, compile=False)

    tokenizer = load_tokenizer_once()

    # Extract features and generate caption
    feature = extract_feature_single(image_path)
    raw_caption = predict_caption(caption_model, feature, tokenizer, MAX_LENGTH)

    # Clean the caption
    clean_result = clean_caption(raw_caption)

    return clean_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('image', help="Path to the input image")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument('--quiet', action='store_true', help="Suppress all output except the final caption")
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    try:
        if not args.quiet:
            print(f"Processing image: {args.image}")

        caption = generate_caption_for_image(args.image)

        if args.quiet:
            print(caption)
        else:
            print("\nGenerated Caption:", caption)
    except Exception as e:
        print(f"Error: {str(e)}")