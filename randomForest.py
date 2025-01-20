import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
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

# # Membuat model Random Forest
# model = RandomForestClassifier(random_state=42, n_estimators = 150, min_samples_leaf=5, min_samples_split=10, max_depth = 25)
# model = RandomForestClassifier(random_state=42, n_estimators = 100, min_samples_leaf=15, min_samples_split=20, max_depth = 10)
model = RandomForestClassifier(random_state=42, n_estimators = 400, min_samples_leaf=3, min_samples_split=6, max_depth = 25)

# # Melatih model dengan data training
model.fit(X_train, y_train)

dataPrediction = [
    [7, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3, 0.45, 8.8],
    [6.6, 0.17, 0.38, 1.5, 0.032, 28, 112, 0.9914, 3.25, 0.55, 11.4],
    [5, 0.55, 0.14, 8.3, 0.032, 35, 164, 0.9918, 3.53, 0.51, 12.5],
    [6.9, 0.36, 0.34, 4.2, 0.018, 57, 119, 0.9898, 3.28, 0.36, 12.7]]

# y_pred = model.predict(dataPrediction)

# # Prediksi hasil untuk data testing
y_pred = model.predict(X_test)

# # Evaluasi model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
# print('Predict:', y_pred)
# print('Classification Report:')
# print(classification_report(y_test, y_pred))
# print('Confusion Matrix:')
# print(confusion_matrix(y_test, y_pred))

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