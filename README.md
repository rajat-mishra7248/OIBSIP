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

* # Task 3: Car Price Prediction with Machine Learning

## 📌 Objective
Build a machine learning regression model to predict the selling price of used cars based on features like car age, brand, kilometers driven, fuel type, and transmission.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook

## 📊 Dataset
* **Source:** Vehicle dataset from Cardekho (`cardataset.csv`)
* **Records:** 4,340 car listings with features such as `name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, and `owner`.

## ⚙️ Workflow
1. **Data Cleaning:** Checked for null values and duplicate records.
2. **Feature Engineering:** Calculated `car_age` from the manufacturing year and extracted the `brand` name from the car title.
3. **Exploratory Data Analysis (EDA):** Visualized selling price distribution, price vs. fuel type box plots, price vs. car age scatter plot, and a feature correlation heatmap.
4. **Encoding & Splitting:** Applied One-Hot Encoding (`pd.get_dummies`) to categorical variables and split the data (80% train, 20% test).
5. **Model Training & Evaluation:** Trained **Linear Regression** and **Random Forest Regressor**, evaluating them with MAE, RMSE, and R² Score.

## 📈 Results

| Model | MAE | RMSE | R² Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 185,156 | 378,944 | 0.5294 |
| **Random Forest Regressor** | 114,953 | 278,436 | 0.7460 |

* **Conclusion:** **Random Forest Regressor** outperformed Linear Regression with an **R² score of 0.7460**. Feature importance analysis showed that `car_age` and `km_driven` are the strongest predictors of car price.


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

