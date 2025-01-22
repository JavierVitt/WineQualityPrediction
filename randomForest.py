import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

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

# # Membuat model Random Forest
# model = RandomForestClassifier(random_state=42, n_estimators = 150, min_samples_leaf=5, min_samples_split=10, max_depth = 25)
# model = RandomForestClassifier(random_state=42, n_estimators = 100, min_samples_leaf=15, min_samples_split=20, max_depth = 10) # Accuracy: 0.8295918367346938
# model = RandomForestClassifier(random_state=42, n_estimators = 400, min_samples_leaf=3, min_samples_split=6, max_depth = 25) # Accuracy: 0.8704081632653061
# model = RandomForestClassifier(random_state=42, n_estimators = 250, min_samples_leaf=10, min_samples_split=10, max_depth = 20) # Accuracy: 0.8425209164125421
# model = RandomForestClassifier(random_state=42, n_estimators = 300, min_samples_leaf=1, min_samples_split=2, max_depth = 20, max_features='sqrt') # Accuracy: 0.8700833398493497
# model = RandomForestClassifier(random_state=42, max_depth = 5, max_features=20, min_samples_leaf=1, min_samples_split=2, n_estimators=85) #nnti otak atik ini lagi
# model = RandomForestClassifier(random_state=42, max_depth = 3, max_features=9, min_samples_leaf=1, min_samples_split=2, n_estimators=110) #ini lebi gg drpd atas
model = RandomForestClassifier(random_state=42, max_depth = 3, max_features=9, min_samples_leaf=1, min_samples_split=2, n_estimators=110, bootstrap=True, n_jobs=-1)
# model = GradientBoostingClassifier(
#     n_estimators=100,       # Jumlah pohon
#     learning_rate=0.1,      # Kecepatan pembelajaran
#     max_depth=3,            # Kedalaman maksimum setiap pohon
#     random_state=42
# )

scores = cross_val_score(model, X_train, y_train, cv=5)  # 5-fold cross-validation

# # Melatih model dengan data training
model.fit(X_train, y_train)

dataPrediction = [
    [7, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3, 0.45, 8.8], # 0
    [6.6, 0.17, 0.38, 1.5, 0.032, 28, 112, 0.9914, 3.25, 0.55, 11.4], # 1
    [5, 0.55, 0.14, 8.3, 0.032, 35, 164, 0.9918, 3.53, 0.51, 12.5], # 1
    [6.9, 0.36, 0.34, 4.2, 0.018, 57, 119, 0.9898, 3.28, 0.36, 12.7], #1
    [7.7, 0.15, 0.29, 1.3, 0.029, 10, 64, 0.9932, 3.35, 0.39, 10.1] # 0
    ]

#Predict: [0,1,1,1,0]

y_pred = model.predict(dataPrediction)

# # Prediksi hasil untuk data testing
# y_pred = model.predict(X_test)

# # Evaluasi model
# print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Predict:', y_pred)
# print('Classification Report:')
# print(classification_report(y_test, y_pred))
# print('Confusion Matrix:')
# print(confusion_matrix(y_test, y_pred))
print(f"Cross-Validation Accuracy: {scores.mean():.2f}")

# Menghitung learning curve
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)

# Menghitung rata-rata dan standar deviasi untuk plot
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

# Plotting
plt.plot(train_sizes, train_mean, label='Akurasi Pelatihan')
plt.plot(train_sizes, test_mean, label='Akurasi Pengujian')
plt.xlabel('Ukuran Data Pelatihan')
plt.ylabel('Akurasi')
plt.legend(loc='best')
plt.title('Learning Curve')
plt.show()

# Best Hyperparameters: {'max_depth': 20, 'min_samples_leaf': 10, 'min_samples_split': 10, 'n_estimators': 250}

