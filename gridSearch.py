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


from sklearn.model_selection import GridSearchCV

# Parameter grid untuk tuning
param_grid = {
    'n_estimators': [100,125,150,175,200,225,250,275,300,325,350,375,400],
    'max_depth': [0,5,10,15,20,25,30,35,40,45,50],
    'min_samples_split': [10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50],
    'min_samples_leaf': [10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50]
}

# GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,
    verbose=1
)

# Menjalankan GridSearch
grid_search.fit(X_train, y_train)

# Output hasil terbaik
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)