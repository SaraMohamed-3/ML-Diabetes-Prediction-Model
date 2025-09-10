# ML-Diabetes-Prediction-Model
This project demonstrates the end-to-end machine learning workflow for predicting diabetes using patient diagnostic data. The dataset used is diabetes.csv.
The goal is to build a predictive system that classifies patients as Diabetic or Non-Diabetic, while practicing data exploration, preprocessing, model training, and deployment steps.

Phase 1: Exploratory Data Analysis (EDA)
Generated summary statistics and checked class balance.
Box Plots: Outlier detection & class comparison.
Bar Plot: Class distribution of target variable (Outcome).
Correlation Heatmap: Detect multicollinearity.

Phase 2: Data Preprocessing
Standardization: All features scaled with StandardScaler (important for SVM and Logistic Regression).
Train/Test Split: 80/20 split with stratification to preserve class balance.

Phase 3: Model Training & Comparison
Implemented and compared three different models:
1-Logistic Regression
Baseline linear model for classification.
Trained on standardized features.
Provided interpretability through feature coefficients.

2-Random Forest
Ensemble-based model with decision trees.
Captured non-linear relationships.
Feature importance scores used for interpretability.

3-Support Vector Machine (SVM)
Tuned using GridSearchCV across hyperparameters:
C (regularization strength)
kernel ( rbf)
gamma 
Achieved high accuracy with strong generalization.

Evaluation Metrics for All Models:
Accuracy
Precision, Recall, F1-score (Classification Report)
Confusion Matrix

Phase 4: Prediction Engine
Built a function predict_diabetes() that:
Accepts new patient data in the same format as the dataset:
[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
Scales input using the training scaler.
Returns prediction: "Diabetic" or "Non-Diabetic".

How to Run:
install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn
Place the diabetes.csv file in the working directory.
Open the Jupyter Notebook and run step by step
