# Task 3: Car Price Prediction with Machine Learning

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
