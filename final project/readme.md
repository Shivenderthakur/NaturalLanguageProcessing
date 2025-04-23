# Fake News Detection Pipeline

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A college-level project demonstrating end-to-end fake news detection using classical and deep learning models. It downloads and preprocesses a public “Fake vs Real News” dataset, explores basic statistics, trains multiple classifiers (Logistic Regression, SVM, Random Forest, MLP), and saves the trained models for inference.

---



## 🚀 Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/Shivenderthakur/NaturalLanguageProcessing.git
cd fake-news-detection
```

### 2. Install dependencies

```bash
pip install -r numpy pandas scikit-learn streamlit nltk 
```

### 3. Download NLTK data

On first run, execute the helper script to download all NLTK corpora:

```
import nltk
nltk.download("all")
```

### 4. Fetch the Dataset

Open the Jupyter notebook:

```
notebooks/preprocessing_and_training_fake_news.ipynb
```

Run the first cell to download and copy the dataset:

```python
import kagglehub
path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
!cp $path/* data/
```

---

## 🧰 Tech Stack & Libraries

- **Python 3.8+**  
- **Data handling**: `pandas`, `numpy`  
- **Text processing**: `nltk` (stopwords, PorterStemmer)  
- **Feature extraction**: `scikit-learn`’s `TfidfVectorizer`  
- **Classical ML models**:  
  - `LogisticRegression`  
  - `SVC` (Support Vector Machine)  
  - `RandomForestClassifier`  
  - `MLPClassifier` (Multi-layer Perceptron)  
- **Deep learning (optional)**: `tensorflow.keras` (Tokenizer, Embedding, LSTM)  
- **Model persistence**: `pickle`, `joblib`  
- **Visualization**: `matplotlib`

---

## 🔍 Key Features

1. **Data Exploration**  
   - Compares unique value counts between fake vs real news.  
   - Visualizes distributions of metadata fields.

2. **Preprocessing Pipeline**  
   - Text cleaning (remove special chars, lowercase).  
   - Tokenization & stemming.  
   - Stop-word removal using NLTK.

3. **Model Training & Evaluation**  
   - Train/Test split of titles dataset.  
   - Trains multiple classifiers on news titles.  
   - Evaluates models and saves the best-performing artifacts.

4. **Inference & Deployment**  
   - Load `title_vectorizer.pkl` + chosen model (e.g., `svm_model.pkl`).  
   - Call `model.predict(vectorizer.transform([new_title]))` for inference.

---

## 📈 Performance Metrics

| Model                  | Accuracy |
|------------------------|---------:|
| Logistic Regression    |   93.59% |
| Support Vector Machine |   94.75% |
| Random Forest          |   92.84% |
| MLP Classifier         |   93.28% |

*Metrics evaluated on held-out test set.*

---

## 📦 Usage Example

```python
import joblib

# Load artifacts
vectorizer = joblib.load("title_vectorizer.pkl")
svm_model   = joblib.load("svm_model.pkl")
rf_model    = joblib.load("rf_model.pkl")
mlp_model   = joblib.load("mlp_model.pkl")
log_model   = joblib.load("logistic_model.pkl")

# Example single prediction
headline = "New breakthrough in renewable energy technology"
X_new    = vectorizer.transform([headline])
print("Prediction:", "Real" if svm_model.predict(X_new)[0] else "Fake")
```

---

## 🔧 Batch Predictions Example

```python
headlines = [
    'Trump reaffirms commitment to defend U.S. and allies',
    'U.S. negotiator says direct diplomacy needed on North Korea',
    "Trump, Mexico's Pena Nieto agree to work out differences on wall",
    "'You're fired' - Trump effigy feels the heat on UK bonfire night",
    'Trump demands U.S. Supreme Court Justice Ginsburg resign over criticism',
    'CATHERINE HERRIDGE: ‘Verifiable Proof’ Comey Drafted Hillary’s Exoneration Letter TWO MONTHS Prior To Interviewing Key Witnesses [Video]',
    ' Hey Conservatives, You Might Want To STFU About ‘Klansman’ Robert Byrd And Clinton',
    'CHARLES BARKLEY DROPS TRUTH-BOMB: Blacks, Not White People Or Cops Are Keeping Blacks Down',
    ' Here Are 12 Tweets Trump DEFINITELY Regrets Sending',
    'BRAVO! COL RALPH PETERS Hammers Obama On Why His Trip To Hiroshima Is So Offensive [Video]'
]

for h in headlines:
    x = vectorizer.transform([h])
    print("===========================================")
    print(h)
    print("SVM Prediction:", "Real" if svm_model.predict(x)[0] else "Fake")
    print("Random Forest Prediction:", "Real" if rf_model.predict(x)[0] else "Fake")
    print("MLP Prediction:", "Real" if mlp_model.predict(x)[0] else "Fake")
    print("Logistic Regression Prediction:", "Real" if log_model.predict(x)[0] else "Fake")
```

### Sample Output

```
===========================================
Trump reaffirms commitment to defend U.S. and allies
SVM Prediction: Real
Random Forest Prediction: Real
MLP Prediction: Real
Logistic Regression Prediction: Real
===========================================
U.S. negotiator says direct diplomacy needed on North Korea
SVM Prediction: Real
Random Forest Prediction: Real
MLP Prediction: Real
Logistic Regression Prediction: Real
===========================================
Trump, Mexico's Pena Nieto agree to work out differences on wall
SVM Prediction: Real
Random Forest Prediction: Real
MLP Prediction: Real
Logistic Regression Prediction: Real
===========================================
'You're fired' - Trump effigy feels the heat on UK bonfire night
SVM Prediction: Real
Random Forest Prediction: Real
MLP Prediction: Real
Logistic Regression Prediction: Real
===========================================
Trump demands U.S. Supreme Court Justice Ginsburg resign over criticism
SVM Prediction: Real
Random Forest Prediction: Real
MLP Prediction: Real
Logistic Regression Prediction: Real
===========================================
CATHERINE HERRIDGE: ‘Verifiable Proof’ Comey Drafted Hillary’s Exoneration Letter TWO MONTHS Prior To Interviewing Key Witnesses [Video]
SVM Prediction: Fake
Random Forest Prediction: Fake
MLP Prediction: Fake
Logistic Regression Prediction: Fake
===========================================
 Hey Conservatives, You Might Want To STFU About ‘Klansman’ Robert Byrd And Clinton
SVM Prediction: Fake
Random Forest Prediction: Fake
MLP Prediction: Fake
Logistic Regression Prediction: Fake
===========================================
CHARLES BARKLEY DROPS TRUTH-BOMB: Blacks, Not White People Or Cops Are Keeping Blacks Down
SVM Prediction: Fake
Random Forest Prediction: Fake
MLP Prediction: Fake
Logistic Regression Prediction: Fake
===========================================
 Here Are 12 Tweets Trump DEFINITELY Regrets Sending
SVM Prediction: Fake
Random Forest Prediction: Fake
MLP Prediction: Fake
Logistic Regression Prediction: Fake
===========================================
BRAVO! COL RALPH PETERS Hammers Obama On Why His Trip To Hiroshima Is So Offensive [Video]
SVM Prediction: Fake
Random Forest Prediction: Fake
MLP Prediction: Fake
Logistic Regression Prediction: Fake
```

---


## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## ✉️ Contact

**SHIVENDER SINGH THAKUR**  
Department of Computer Science, M.B.M University
✉️ shivthakur4512@gmail.com
