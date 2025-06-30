# 🎵 Music Genre Prediction with PCA and Logistic Regression

This project explores and builds a machine learning pipeline to predict music genres using logistic regression. It includes data preprocessing, exploratory data analysis (EDA), dimensionality reduction using Principal Component Analysis (PCA), and genre classification. It also compares the model performance using both the original and PCA-transformed feature spaces.

---

## 📊 Features

- Exploratory Data Analysis (EDA) on music metadata
- Missing value handling
- Label encoding for genre classification
- Correlation analysis between features
- Dimensionality reduction using PCA
- Model training with Logistic Regression
- Performance comparison (PCA vs. Original features)
- Genre prediction for tracks with missing labels

---

## 🧪 Tech Stack

- **Python 3.x**
- **pandas**, **numpy** – data manipulation
- **seaborn**, **matplotlib** – visualization
- **scikit-learn** – machine learning (PCA, logistic regression, evaluation)

---

## 📁 Dataset

The model uses a dataset called `music_dataset_mod.csv`, which should contain music track features and a `Genre` column. 


---



## 📈 Model Evaluation

The script compares classification accuracy between:
- Original standardized feature space
- PCA-reduced feature space

It also prints:
- Accuracy scores
- Classification reports (precision, recall, F1-score)

---

## 🔍 Genre Prediction

The model detects songs with missing genre labels and predicts their genres using the trained PCA-based model.

---

## 🧠 Future Improvements

- Use other classifiers (Random Forest, SVM, etc.)
- Perform grid search for hyperparameter tuning
- Add interactive dashboard (e.g., Streamlit or Dash)
- Visualize PCA components

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


