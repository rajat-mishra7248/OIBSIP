# Task 4: Email Spam Detection with Machine Learning

## 📌 Objective
Build a Natural Language Processing (NLP) binary classification model to distinguish between **spam** and **legitimate (ham)** messages.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn (TF-IDF, Naive Bayes, Logistic Regression), Regex (`re`), Matplotlib, Seaborn
* **Environment:** Jupyter Notebook

## 📊 Dataset
* **Source:** SMS Spam Collection Dataset (`spamdataset.csv`)
* **Records:** 5,572 messages labeled as `ham` (4,825) or `spam` (747).

## ⚙️ Workflow
1. **Data Cleaning:** Removed unnecessary unnamed columns, standardized labels, and checked for missing values.
2. **Text Preprocessing:** Converted text to lowercase, removed punctuation/special characters, and stripped extra whitespace.
3. **Feature Extraction:** Used `TfidfVectorizer(stop_words="english")` to convert cleaned text into numerical feature vectors.
4. **Train/Test Split:** Split the dataset into 80% training and 20% testing sets using stratified sampling.
5. **Model Training & Evaluation:** Trained **Multinomial Naive Bayes** and **Logistic Regression**, evaluating them using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 96.68% | 99.12% | 75.84% | 0.8593 |
| **Logistic Regression** | 96.68% | 100.00% | 75.17% | 0.8582 |

* **Conclusion:** Both models achieved **96.68% accuracy**. **Multinomial Naive Bayes** achieved a slightly higher F1-score (0.8593) and recall (75.84%), making it highly effective for catching spam while avoiding false positives.
