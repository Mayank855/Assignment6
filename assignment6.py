import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import randint
import warnings
warnings.filterwarnings("ignore")

# load data
data = load_breast_cancer()
X = data.data
y = data.target

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# models
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB()
}

# evaluation function
def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall   :", round(recall_score(y_test, y_pred), 4))
    print("F1-score :", round(f1_score(y_test, y_pred), 4))

# train and test all models
print("Initial Model Evaluation")
for name, model in models.items():
    evaluate_model(name, model)

# hyperparameter tuning - SVM
svc_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
grid_svc = GridSearchCV(SVC(), svc_params, cv=5, scoring='f1')
grid_svc.fit(X_train, y_train)
print("\nBest SVC Params:", grid_svc.best_params_)
evaluate_model("Tuned SVC", grid_svc.best_estimator_)

# hyperparameter tuning - Random Forest
rf_params = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(2, 10),
    'min_samples_split': randint(2, 10)
}
random_rf = RandomizedSearchCV(RandomForestClassifier(), rf_params, n_iter=10, cv=5, scoring='f1', random_state=42)
random_rf.fit(X_train, y_train)
print("\nBest RF Params:", random_rf.best_params_)
evaluate_model("Tuned Random Forest", random_rf.best_estimator_)

# final comparison
final_models = {
    "Logistic Regression": LogisticRegression(),
    "Tuned SVC": grid_svc.best_estimator_,
    "Tuned Random Forest": random_rf.best_estimator_
}

results = []

for name, model in final_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    })

df_results = pd.DataFrame(results)
print("\nFinal Comparison Table:\n")
print(df_results.sort_values(by="F1-score", ascending=False).to_string(index=False))
