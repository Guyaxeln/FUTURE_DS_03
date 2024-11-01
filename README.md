Loan Approval Prediction Model
A machine learning project to predict whether a loan application will be approved based on various applicant factors such as income, loan amount, credit history, etc. This project uses Logistic Regression and Random Forest Classifier to create a binary classification model, trained on a sample dataset of loan applications.

Table of Contents
Project Overview
Dataset
Installation
Usage
Results
Model Export
Contributing
License

Project Overview
The goal of this project is to predict the approval status of loan applications using supervised machine learning techniques. The model will classify applications as either "Approved" or "Rejected" based on historical data, providing valuable insights for the loan approval process.

Dataset
The dataset used for this project includes various features, such as:

ApplicantIncome: Applicant's income
CoapplicantIncome: Income of co-applicant (if any)
LoanAmount: Total loan amount requested
Loan_Amount_Term: Loan term in months
Credit_History: Credit history of the applicant (1: good, 0: bad)
Dependents: Number of dependents of the applicant
Education: Education level of the applicant (Graduate/Not Graduate)
Property_Area: Area type where the property is located (Urban/Semiurban/Rural)
Sample Dataset
The dataset used for this model can be obtained here.

Installation
Clone the repository:

bash
Copier le code
git clone https://github.com/Guyaxeln/FUTURE_DS_03.git
   cd FUTURE_DS_03
Install the required packages:

bash
Copier le code
pip install -r requirements.txt
Usage
To train the models and make predictions, follow these steps:

Train the Logistic Regression Model:

python
Copier le code
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train, y_train)
Train the Random Forest Classifier:

python
Copier le code
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
Make Predictions:

python
Copier le code
y_pred_lr = model_lr.predict(X_test)
y_pred_rf = model_rf.predict(X_test)
Evaluate the Models: Use the included code for model evaluation, such as accuracy score and confusion matrix.

Export the Models:

python
Copier le code
import joblib
joblib.dump(model_lr, 'logistic_regression_model.joblib')
joblib.dump(model_rf, 'random_forest_model.joblib')
Visualization
To visualize the confusion matrix and feature importances:

python
Copier le code
# Confusion Matrix for Logistic Regression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

# Feature Importance for Random Forest
importances = model_rf.feature_importances_
feature_names = X.columns
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance for Random Forest')
plt.show()

Results
The models provide predictions for loan approvals with accuracies of approximately XX% for Logistic Regression and YY% for Random Forest on the test data (example values; actual results may vary). These models can assist financial institutions in making preliminary loan approval decisions more effectively.

Model Export
The trained models can be exported and loaded as follows:

python
Copier le code
# Exporting the models
joblib.dump(model_lr, 'logistic_regression_model.joblib')
joblib.dump(model_rf, 'random_forest_model.joblib')

# Loading the models
loaded_model_lr = joblib.load('logistic_regression_model.joblib')
loaded_model_rf = joblib.load('random_forest_model.joblib')
y_pred_loaded_lr = loaded_model_lr.predict(X_test)
y_pred_loaded_rf = loaded_model_rf.predict(X_test)
Contributing
Contributions are welcome! If you’d like to make improvements, please fork the repository and submit a pull request.

License
This project is licensed under the MIT License.
