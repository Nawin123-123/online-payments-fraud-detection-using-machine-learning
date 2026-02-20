Online Payments Fraud Detection ML Model
Fraud Detection Python XGBoost AUC ROC License

Project Overview
The Online Payments Fraud Detection application is designed to predict fraudulent transactions in online payment systems using advanced machine learning techniques. With the growing risk of online payment fraud, this model helps financial institutions and e-commerce platforms identify suspicious transactions in real-time.

Our solution leverages machine learning models, such as XGBoost, to detect fraudulent transactions based on a set of features that describe the transaction details. This project focuses on:

Real-time fraud detection by integrating machine learning models into the payment processing system.
Interactive interface for users to input transaction details and receive immediate feedback on transaction validity.
Comprehensive feature engineering and data reduction techniques to optimize model performance.
Features
Fraud Detection Model: Built with XGBoost and tuned using hyperparameter optimization to achieve an AUC ROC score of 0.9556.
Feature Reduction: Dimensionality reduction from 394 to 53 features for optimized performance.
Interactive Web Application: A user-friendly interface built with Streamlit for real-time fraud prediction.
Scalable: Suitable for integration into large-scale online payment platforms.
Visualization Tools: Displays transaction analysis and prediction results in an intuitive format.
Comprehensive Documentation: Access detailed documentation about what has been done through the links provided below.
Usage
Open the web app by running the command below.

https://online-payment-fraud-detector.streamlit.app/

Enter transaction details like:

Transaction amount
Transaction ID
Device Type
Card Type (Visa/ discover/ american express/ mastercard)
Transaction type (debit/credit)
Click the Predict button to determine if the transaction is fraudulent.

View the prediction result (fraudulent or genuine).

Model Pipeline
The machine learning pipeline for this project consists of the following steps:

Data Preprocessing: Handling missing values, scaling, and encoding categorical features.
Feature Engineering: Reducing the dataset from 394 features to 53 using techniques like PCA and feature importance.
Model Training: Using XGBoost with hyperparameter tuning for optimal performance.
Model Evaluation: Evaluating the model based on metrics such as AUC ROC and accuracy.
Prediction: Real-time predictions using the trained model.
NOTE: To dive deeper into the technical details, model architecture, or other functionalities, visit the following sections:

NOTEBOOK LINK: https://www.kaggle.com/code/avdhesh15/fraud-detection-model
WEBSITE LINK: https://avdhesh-varshney.github.io/online-payment-fraud-detection-app/

Evaluation Metrics
The model was evaluated on the IEEE-CIS Fraud Detection Dataset using the following metrics:

1 CV Score: 0.9562
2 CV Score: 0.952
3 CV Score: 0.9586
Mean AUC ROC: 0.9556
These metrics ensure a reliable detection of fraudulent transactions, minimizing both false positives and false negatives.

Dataset
This project uses the IEEE-CIS Fraud Detection Dataset, which contains anonymized transaction data. The dataset includes a wide range of features that describe each transaction, such as transaction amount, device information, and transaction type.

Training Data: Used for model training and validation.
Testing Data: Used to evaluate the final model.
Technologies Used
Python 3.8+
XGBoost: The primary machine learning model used for prediction.
Streamlit: For building the web interface and API.
Pandas, NumPy: For data manipulation and preprocessing.
Scikit-learn: For preprocessing and model evaluation.
Matplotlib, Seaborn: For visualizations and feature analysis.
Show some  ❤️  by  🌟  this repository!
