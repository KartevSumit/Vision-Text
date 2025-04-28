import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.translate.bleu_score import corpus_bleu

# Import from our modules
from feature_extraction import extract_features, load_features
from model import create_caption_model, data_generator, predict_caption

# Configuration
BASE_DIR = 'D:/Program/archive'
WORKING_DIR = './Data'
EPOCHS = 50
BATCH_SIZE = 32


def load_captions(captions_path):
    with open(captions_path, 'r') as f:
        next(f)  # Skip header
        captions_doc = f.read()

    mapping = {}

    for line in tqdm(captions_doc.split('\n')):
        tokens = line.split('.')
        if len(line) < 2:
            continue

        image_id, caption = tokens[0], tokens[1]
        image_id = image_id.split('.')[0]
        caption = caption.strip()

        if image_id not in mapping:
            mapping[image_id] = []

        mapping[image_id].append(caption)

    return mapping


def clean_captions(mapping):
    """Clean and preprocess captions"""
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            # Convert to lowercase
            caption = caption.lower()
            # Remove non-alphabetic characters
            caption = ''.join([c for c in caption if c.isalpha() or c.isspace()])
            # Remove extra spaces
            caption = ' '.join(caption.split())
            # Add start and end tokens
            caption = 'startseq ' + ' '.join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption


def main():
    # Extract or load image features
    features_path = os.path.join(WORKING_DIR, 'features.pkl')
    if os.path.exists(features_path):
        print("Loading pre-extracted features...")
        features = load_features(features_path)
    else:
        print("Extracting image features...")
        features = extract_features(
            os.path.join(BASE_DIR, 'Images'),
            features_path
        )

    # Load and process captions
    print("Loading and processing captions...")
    mapping = load_captions(os.path.join(BASE_DIR, 'captions.txt'))
    clean_captions(mapping)

    # Prepare data for model
    print("Preparing data for model training...")
    all_captions = []
    for key in mapping:
        all_captions.extend(mapping[key])

    # Create tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")

    # Calculate maximum caption length
    max_length = max(len(caption.split()) for caption in all_captions)
    print(f"Maximum caption length: {max_length}")

    # Split data into training and testing sets
    image_ids = list(mapping.keys())
    split = int(len(image_ids) * 0.90)
    train_ids = image_ids[:split]
    test_ids = image_ids[split:]
    print(f"Training samples: {len(train_ids)}, Testing samples: {len(test_ids)}")

    # Check if model already exists
    model_path = os.path.join(WORKING_DIR, 'best_model.h5')

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
    else:
        # Create and train model
        print("Creating model...")
        model = create_caption_model(vocab_size, max_length)

        print("Training model...")
        steps_per_epoch = len(train_ids)
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            generator = data_generator(
                train_ids, mapping, features, tokenizer,
                max_length, vocab_size, BATCH_SIZE
            )
            model.fit(generator, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)

        # Save the trained model
        print(f"Model saved to {model_path}")
        model.save(model_path)

    # Evaluate model using BLEU score
    print("Evaluating model...")
    actual, predicted = list(), list()

    batch_size = 25
    for i in range(0, len(test_ids), batch_size):
        batch_ids = test_ids[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(test_ids) - 1) // batch_size + 1}")

        for key in tqdm(batch_ids):
            captions = mapping[key]
            y_pred = predict_caption(model, features[key], tokenizer, max_length)

            actual_captions = [caption.split() for caption in captions]
            y_pred = y_pred.split()

            actual.append(actual_captions)
            predicted.append(y_pred)

        import gc
        gc.collect()

    # Calculate BLEU scores
    print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))


if __name__ == "__main__":
    main()