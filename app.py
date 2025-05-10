from flask import Flask, request, jsonify
import torch
import json
import os
from flask_cors import CORS

from fun import TransformerModel, PositionalEncoding, translate_text_to_gloss, preprocess_text

app = Flask(__name__)
CORS(app)
# --- Model and Vocabulary Loading ---
model_path = ''
vocab_file_path = os.path.join(model_path, 'transformer_model.pt.vocab.json')
model_weights_path = os.path.join(model_path, 'transformer_model.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(vocab_file_path, 'r') as f:
    vocab_data = json.load(f)
text_vocab = vocab_data['text_vocab']
gloss_vocab = vocab_data['gloss_vocab']
text_word_to_index = {token: idx for idx, token in enumerate(text_vocab)}
text_index_to_word = {idx: token for idx, token in enumerate(text_vocab)}
gloss_word_to_index = {token: idx for idx, token in enumerate(gloss_vocab)}
gloss_index_to_word = {idx: token for idx, token in enumerate(gloss_vocab)}

# --- Instantiate and Load Model ---
EMBEDDING_DIM = 1024
NHEAD = 8
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DROPOUT = 0.1
MAX_LEN = 100

model = TransformerModel(
    len(text_vocab),
    len(gloss_vocab),
    EMBEDDING_DIM,
    NHEAD,
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    DROPOUT,
    MAX_LEN
).to(device)

model.load_state_dict(torch.load(model_weights_path, map_location=device)['model_state_dict'])
model.eval()
print("Model loaded in Flask app!")

@app.route('/hello')
def home():
    return "Text-to-Gloss Translation Server is running!"

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    text_to_translate = data.get('text', '') # Get text from JSON request

    if not text_to_translate:
        return jsonify({'error': 'No text provided'}), 400 # Bad request

    predicted_gloss = translate_text_to_gloss(
        model, preprocess_text(text_to_translate), text_word_to_index, gloss_word_to_index, gloss_index_to_word, device
    )
    return jsonify({'gloss': predicted_gloss}) # Return gloss as JSON


if __name__ == '__main__':
    app.run(debug=True) # Run in debug mode for development
