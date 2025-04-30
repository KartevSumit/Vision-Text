# Vision to Voice: A Data Science Approach to Image Captioning for the Visually Impaired

A deep learning system that generates descriptive captions for images and converts them into spoken audio using offline text-to-speech.

## Overview

This project implements an end-to-end neural image captioning system with integrated text-to-speech functionality. It combines Convolutional Neural Networks (CNNs) for image feature extraction with Recurrent Neural Networks (RNNs) for sequence generation. Generated captions are then converted to speech using an offline TTS engine.

## Technical Architecture

### Image Feature Extraction

- Uses a pre-trained VGG16 model (excluding the classification head)
- Extracts 4096-dimensional feature vectors from images
- Caches extracted features for reuse

```python
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

def extract_features(img_path):
    model = VGG16(include_top=False, pooling='avg')
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features
```

### Caption Generation

- Encoder-decoder architecture using LSTM
- Merges image feature vectors with text embeddings
- Uses teacher forcing during training
- Outputs next word prediction using a softmax classifier

```python
def create_caption_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model
```

### Text-to-Speech (TTS)

- Uses `pyttsx3` for offline speech synthesis
- Optionally supports MP3 output using `pydub` and `ffmpeg`
- Compatible with Windows, macOS, and Linux

```python
import pyttsx3

def speak_caption(text, output_path=None):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    if output_path:
        engine.save_to_file(text, output_path)
        engine.runAndWait()
    else:
        engine.say(text)
        engine.runAndWait()
```

## Project Structure

```
neural-image-captioning/
├── main.py                 # Training script
├── feature_extraction.py   # Extract features using VGG16
├── model.py                # Captioning model architecture
├── get_caption.py          # Caption generation script
├── Text-to-speech.py       # Caption to speech conversion
└── Data/                   # Stored features and trained model
```

## Dataset

This project uses the Flickr8k dataset from Kaggle, which contains 8,000 images each paired with 5 captions (40,000 captions total).

- **Source**: https://www.kaggle.com/datasets/adityajn105/flickr8k
- **Images**: JPG format, 8000 images
- **Captions**: 5 human-written captions per image, provided in a text file

## Requirements

- Python 3.6 or later
- TensorFlow 2.x
- Keras
- NumPy
- NLTK (for BLEU evaluation)
- tqdm
- pyttsx3 (for TTS)
- Optional: pydub and ffmpeg (for MP3 support)

## Installation

1. Clone the repository:

```bash
git https://github.com/KartevSumit/Vision-Text.git
cd Vision-Text
```

2. Install the dependencies:

```bash
pip install tensorflow keras numpy nltk tqdm pyttsx3
pip install pydub  # optional
```

3. Ensure ffmpeg is installed and added to your system PATH for MP3 support.

4. Download or generate model files:
   - Run training to generate your own model, or  
   - Download pre-trained files from [Link](https://drive.google.com/drive/folders/1gDGfJZMy3U2sh1Uf9t4H0JkjRzk7dvYE?usp=sharing)

Expected directory structure:

```
./Data/
├── features.pkl    # Precomputed image features (~1.2GB)
└── best_model.h5   # Trained model weights (~150MB)
```

## Usage

### Train the Model

```bash
python main.py
```

- Extracts features from images
- Processes and tokenizes captions
- Trains the LSTM-based model
- Evaluates performance using BLEU scores
- Saves the best model

### Generate Captions

```bash
python get_caption.py path/to/image.jpg
```

Optional: specify GPU device

```bash
python get_caption.py path/to/image.jpg --gpu_id 0
```

### Generate Speech Output

```bash
python Text-to-speech.py path/to/image.jpg
```

To save audio to a file:

```bash
python Text-to-speech.py path/to/image.jpg --save output.wav
```

## Technical Details

- Captions are lowercased, cleaned, and tokenized
- Special tokens `startseq` and `endseq` mark sentence boundaries
- Images resized to `224 x 224` for VGG16
- Training uses mini-batches, categorical crossentropy, and Adam optimizer
- Caption generation uses greedy search decoding
- GPU memory growth enabled to prevent OOM errors
- Garbage collection and model reuse reduce memory overhead

## Evaluation

The system uses BLEU scores to measure the quality of generated captions by comparing them against reference descriptions. Current performance on the Flickr8k test split:

- BLEU-1: ~0.52
- BLEU-2: ~0.31

## License

This project is licensed under the MIT License.

## Acknowledgments

- VGG16 pre-trained weights from ImageNet
- Libraries: TensorFlow, Keras, pyttsx3

