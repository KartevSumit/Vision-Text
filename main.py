import os
from config import FEATURES_FILE, CAPTIONS_FILE, BASE_DIR
from preprocess import extract_image_features, load_features, load_captions, create_tokenizer, max_caption_length
from train_and_eval import train_model, evaluate_model


def main():
    # 1. Features
    if not os.path.exists(FEATURES_FILE):
        features = extract_image_features(BASE_DIR, FEATURES_FILE)
    else:
        features = load_features(FEATURES_FILE)

    # 2. Captions & Tokenizer
    captions   = load_captions(CAPTIONS_FILE)
    tokenizer  = create_tokenizer(captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_len    = max_caption_length(captions)

    # 3. Split
    ids       = list(captions.keys())
    split     = int(len(ids) * 0.9)
    train_ids = ids[:split]
    test_ids  = ids[split:]

    # 4. Train & Evaluate
    train_model(train_ids, captions, features, tokenizer, max_len, vocab_size)
    evaluate_model(test_ids, captions, features, tokenizer, max_len)

if __name__ == '__main__':
    main()