import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('wine.csv', delimiter=';')

# Melihat beberapa baris pertama dari dataset
# print(df.head())

# Statistik deskriptif
# print(df.describe())

# Cek apakah ada nilai yang hilang
# print(df.isnull().sum())

# Visualisasi hubungan antara fitur dan target (quality)
# plt.figure(figsize=(10,6))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap')
# plt.show()

# Pisahkan fitur dan target
X = df.drop('quality', axis=1)  # Fitur
y = df['quality']  # Target

# Bagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model Linear Regression
model = LinearRegression()

# Melatih model
model.fit(X_train, y_train)

# Prediksi pada data testing
y_pred = model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")

# Plot perbandingan antara nilai aktual dan prediksi
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.title('Prediksi vs Aktual')
plt.xlabel('Aktual')
plt.ylabel('Prediksi')
plt.show()

