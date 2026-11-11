# OIBSIP
# Task 1: Iris Flower Classification

## 📌 Objective
Build a machine learning classification model to accurately identify the species of an Iris flower (**Setosa**, **Versicolor**, or **Virginica**) based on its sepal and petal measurements.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook

## 📊 Dataset
* **Source:** Built-in Iris dataset from `sklearn.datasets.load_iris()`
* **Features:** Sepal Length, Sepal Width, Petal Length, Petal Width (150 samples across 3 classes)

## ⚙️ Workflow
1. **Exploratory Data Analysis (EDA):** Checked dataset shape, data types, missing values, and summary statistics.
2. **Data Visualization:** Created pairplots and box plots to analyze feature distributions across species. Petal length and petal width proved to be the most discriminative features.
3. **Model Training:** Performed an 80/20 train-test split and trained **Logistic Regression** and **K-Nearest Neighbors (KNN)** classifiers.
4. **Evaluation:** Evaluated models using Accuracy Score, Classification Report, and Confusion Matrix.

## 📈 Results
* **Logistic Regression Accuracy:** 100.00%
* **K-Nearest Neighbors (KNN) Accuracy:** 100.00%
* **Conclusion:** Both models achieved 100% accuracy on the test set. **Logistic Regression** is selected as the best model due to its simplicity and fast inference speed.
