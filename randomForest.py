import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
model = RandomForestClassifier(random_state=42, n_estimators = 100, min_samples_leaf=7, min_samples_split=12, max_depth = 25)

# # Melatih model dengan data training
model.fit(X_train, y_train)

# # Prediksi hasil untuk data testing
y_pred = model.predict(X_test)

# # Evaluasi model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

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