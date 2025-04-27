import os
from config import WORK_DIR
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.translate.bleu.score import corpus_bleu
from preprocess import (
    extract_image_features, load_features, load_captions,
    create_tokenizer, max_caption_length
)
from model import define_model, generate_caption


def data_generator(image_ids, captions_map, features, tokenizer, max_len, vocab_size, batch_size):
    """Yield batches ([features, seq_input], next_word)."""
    X1, X2, y = [], [], []
    count = 0
    while True:
        for img_id in image_ids:
            for cap in captions_map[img_id]:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    in_seq, out_word = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
                    out_seq = to_categorical([out_word], num_classes=vocab_size)[0]
                    X1.append(features[img_id][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            count += 1
            if count == batch_size:
                yield [np.array(X1), np.array(X2)], np.array(y)
                X1, X2, y = [], [], []
                count = 0


def train_model(train_ids, captions_map, features, tokenizer, max_len, vocab_size,
                epochs=20, batch_size=32):
    """Train the model and save best weights."""
    model = define_model(max_len, vocab_size)
    steps = len(train_ids)
    for _ in range(epochs):
        gen = data_generator(train_ids, captions_map, features,
                             tokenizer, max_len, vocab_size, batch_size)
        model.fit(gen, epochs=1, steps_per_epoch=steps)
    model.save(os.path.join(WORK_DIR, 'best_model.h5'))


def evaluate_model(test_ids, captions_map, features, tokenizer, max_len):
    """Evaluate the model using BLEU scores."""
    model = load_model(os.path.join(WORK_DIR, 'best_model.h5'))
    actual, predicted = [], []
    for img_id in test_ids:
        real_caps = [c.split() for c in captions_map[img_id]]
        yhat = generate_caption(model, features[img_id], tokenizer, max_len).split()
        actual.append(real_caps)
        predicted.append(yhat)
    print('BLEU-1:', corpus_bleu(actual, predicted, weights=(1, 0, 0, 0)))
    print('BLEU-2:', corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))