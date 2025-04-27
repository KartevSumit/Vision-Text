import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences


def define_model(max_len, vocab_size):
    """Build and compile the captioning model."""
    # Image branch
    inp1 = Input(shape=(4096,))
    fe = Dropout(0.4)(inp1)
    fe = Dense(256, activation='relu')(fe)
    # Text branch
    inp2 = Input(shape=(max_len,))
    se = Embedding(vocab_size, 256, mask_zero=True)(inp2)
    se = Dropout(0.4)(se)
    se = LSTM(256)(se)
    # Merge
    merged = add([fe, se])
    merged = Dense(256, activation='relu')(merged)
    outputs = Dense(vocab_size, activation='softmax')(merged)
    model = Model(inputs=[inp1, inp2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def generate_caption(model, feature, tokenizer, max_len):
    """Generate a caption for one image feature using startseq/endseq tokens."""
    in_text = 'startseq'
    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len)
        yhat = model.predict([feature, seq], verbose=0)
        word_idx = np.argmax(yhat)
        word = idx_to_word(word_idx, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq ', '').replace(' endseq', '')


def idx_to_word(index, tokenizer):
    """Map an integer back to its word."""
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word
    return None
