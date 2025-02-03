
# AI News Validator 📰🤖  

AI News Validator is an **AI-powered web application** that detects **fake news** using **BERT embeddings** and **machine learning** models. In the age of misinformation, this project provides an **efficient** and **accurate** solution to verify news authenticity.  

## ✨ Features  

✅ **Fake News Detection** – Uses **BERT embeddings** and a trained ML model to classify news as real or fake.  
✅ **Deep Learning-Powered** – Utilizes **BERT (Bidirectional Encoder Representations from Transformers)** for advanced NLP processing.  
✅ **User-Friendly Interface** – A simple **Flask-based web app** for real-time news validation.  
✅ **TF-IDF & BERT Embeddings** – Uses a hybrid approach of **traditional NLP (TF-IDF)** and **deep learning (BERT embeddings)** for feature extraction.  
✅ **Model Training & Deployment** – Pretrained BERT combined with **Logistic Regression/SVM** for classification.  

---

## 🚀 Technologies Used  

- **Python** 🐍  
- **Flask** 🌐 (for web framework)  
- **PyTorch / TensorFlow** 🔥 (for BERT embeddings)  
- **Hugging Face Transformers** 🤗 (for BERT model)  
- **Scikit-Learn** 🤖 (for ML model training)  
- **TF-IDF Vectorization** 📊 (for feature extraction)  
- **Pickle** 📦 (for model serialization)  
- **HTML, CSS, JavaScript** 🎨 (for frontend)  

---

## 📥 Installation & Setup  

Follow these steps to set up and run the AI News Validator locally:  

### 1️⃣ Clone the Repository  

```bash
git clone https://github.com/TaneshG13/AI-News-Validator.git
cd AI-News-Validator
```

### 2️⃣ Install Dependencies  

Make sure you have **Python 3.7+** installed. Then, install required dependencies:  

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application  

Start the Flask web server:  

```bash
python app.py
```

Now, open your browser and navigate to **`http://127.0.0.1:5000/`** to access the application.  

---

## 🛠️ Project Structure  

```
AI-News-Validator/
├── static/                 # Contains static assets (CSS, JavaScript, images)
├── templates/              # HTML templates for web pages
├── app.py                  # Main Flask application
├── model_training.ipynb    # Jupyter Notebook for model training
├── fake_news_bert.pkl      # Trained BERT-based ML model
├── fake_news_tfidf.pkl     # Trained TF-IDF-based ML model
├── tfidf_vectorizer.pkl    # TF-IDF vectorizer for text preprocessing
├── bert_tokenizer.pkl      # BERT tokenizer for text processing
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🧠 How It Works  

1️⃣ **Text Preprocessing** – Clean text by removing stopwords, punctuation, and special characters.  
2️⃣ **Feature Extraction**:  
   - **TF-IDF Embeddings** – Converts text into a sparse numerical feature representation.  
   - **BERT Embeddings** – Extracts deep semantic meaning using **Hugging Face’s BERT model**.  
3️⃣ **Model Training** – Two models are trained:  
   - **Logistic Regression (TF-IDF-based)** – Lightweight and efficient.  
   - **BERT + SVM/Logistic Regression** – High accuracy using deep embeddings.  
4️⃣ **Prediction** – The trained model predicts whether a given article is **real** or **fake**.  
5️⃣ **User Interface** – Users can enter a news article in the web app, and the AI provides an instant validation result.  

---

## 📊 Model Training  

The model is trained using the **Fake News Dataset** from Kaggle. The steps include:  

- **Text Preprocessing** – Tokenization, stemming, and removing stopwords.  
- **TF-IDF Feature Extraction** – Converts text into numerical vectors.  
- **BERT Embeddings** – Uses **pretrained BERT** to extract contextualized word representations.  
- **Model Selection** – **Logistic Regression (TF-IDF)** and **SVM (BERT Embeddings)** for classification.  
- **Model Evaluation** – Achieving **high accuracy (~95%)** using a hybrid approach.  

For more details, check **`model_training.ipynb`**.  

---

## 🤝 Contributing  

Contributions are welcome! If you'd like to improve the project, please **fork the repository**, make changes, and submit a **pull request**. You can also open issues for bug reports or feature requests.  

---

## 📩 Contact  

💡 Have questions or suggestions? Feel free to reach out!  

📧 **Email:** taneshg13@gmail.com  
🌐 **GitHub:** [TaneshG13](https://github.com/TaneshG13)  

---

### 🔥 AI-Powered News Verification – Stay Informed, Stay Smart! 🚀  
