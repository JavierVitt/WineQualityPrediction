from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import Lasso

# Load dataset
data = pd.read_csv('wine.csv', delimiter=';')

# Preprocessing: Mengubah target (quality) menjadi 2 kelas: Kualitas tinggi (>=7) dan rendah (<7)
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 7 else 0)  # 1: kualitas tinggi, 0: kualitas rendah

X = data.drop('quality', axis=1)  # Fitur (input)
y = data['quality']  # Target (output)

# Split data menjadi training dan testing set (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = RandomForestClassifier(random_state=42)

# # Melatih model dengan data training
model.fit(X_train, y_train)

# # Prediksi hasil untuk data testing
y_pred = model.predict(X_test)


# Grid Hyperparameter
# param_grid = {
#     'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500],
#     'min_samples_leaf': [5, 7, 10, 12, 15, 18, 20],
#     'min_samples_split': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
#     'max_depth': [20, 25, 30]
# }

# # Grid Search dengan 3-fold CV
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# Hasil terbaik
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)
