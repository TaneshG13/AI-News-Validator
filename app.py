import os
import re
import string
import pickle
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer, BertModel
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

with open('fake_news_clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.to(device)
bert_model.eval()

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def get_batch_bert_embeddings(texts, batch_size=1):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
        torch.cuda.empty_cache()
    return np.vstack(embeddings)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    raw_text = data['text']
    processed_text = clean_text(raw_text)
    tfidf_feat = tfidf_vectorizer.transform([processed_text]).toarray()
    bert_feat = get_batch_bert_embeddings([processed_text], batch_size=1)
    combined_feat = np.hstack([tfidf_feat, bert_feat])
    prediction = clf.predict(combined_feat)[0]
    label = 'real' if prediction == 1 else 'fake'
    return jsonify({'prediction': label})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
