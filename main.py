import os
import pickle
import numpy as np
from tqdm import tqdm

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

# Load the model
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
print(model.summary())

BASE_DIR = 'D:/Program/archive'
WORKING_DIR = 'D:/Program/Data'

features = {}  # Initialize as dictionary, not tuple
directory = os.path.join(BASE_DIR, 'Images')
for img_name in tqdm(os.listdir(directory)):
    img_path = os.path.join(directory, img_name)  # Use os.path.join for paths
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    image_id = img_name.split('.')[0]
    features[image_id] = feature

pickle.dump(features, open(os.path.join(WORKING_DIR,'features.pk1'),'wb'))

with open(os.path.join(WORKING_DIR,'features.pk1'),'rb') as f:
    features = pickle.load(f)

with open(os.path.join(BASE_DIR,'caption.txt'),'r') as f:
    next(f)
    captions_doc = f.read()

mapping=()

for line in tqdm(captions_doc.split('\n')):
    tokens = line.split('.')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1]
    image_id = image_id.split('.')[0]
    caption = ' '.join(caption)
    if image_id not in mapping:
        mapping[image_id] = []

    mapping[image_id].append(caption)

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = caption.replace('[^A-Za-z]', '')
            caption = caption.replace('/s+', ' ')
            caption = '<start >' + " ".join([word for word in caption.split() if len(word)>1]) + ' <end>'
            captions[i] = caption

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocal_size = len(tokenizer.word_index)+1

max_length = max(len(caption.split()) for caption in all_captions)

image_ids = list(mapping.keys())
split = int(len(image_ids)*0.90)
train = image_ids[:split]
test = image_ids[split:]

