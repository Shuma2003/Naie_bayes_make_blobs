# Импортируем библиотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Создаем красивые данные как make_blobs
X, y = make_blobs(n_samples=300, centers=3, n_features=2, 
                  cluster_std=1.0, random_state=42)

print("Данные X:")
print(X[:5])
print("\nМетки y:")
print(y[:10])

# 1. Визуализация данных
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
plt.title('Кластеры данных - Make Blobs')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.colorbar(label='Класс')
plt.grid(True, alpha=0.3)
plt.show()

# 2. Гистограммы признаков
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(X[:, 0], bins=20, alpha=0.7, color='skyblue')
plt.title('Распределение Признака 1')
plt.xlabel('Значение')
plt.ylabel('Частота')

plt.subplot(1, 2, 2)
plt.hist(X[:, 1], bins=20, alpha=0.7, color='lightcoral')
plt.title('Распределение Признака 2')
plt.xlabel('Значение')
plt.ylabel('Частота')

plt.tight_layout()
plt.show()

# 3. Gaussian Naive Bayes
gaussian_nb = GaussianNB()
gaussian_nb.fit(X, y)

# Визуализация границ классификации
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z = gaussian_nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, edgecolors='black')
plt.title('Границы классификации GaussianNB')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.colorbar(label='Класс')
plt.show()

# 4. Предсказания и метрики
y_pred = gaussian_nb.predict(X)

print("Матрица ошибок:")
print(confusion_matrix(y, y_pred))
print('Accuracy:', accuracy_score(y, y_pred))
print('F1 score:', f1_score(y, y_pred, average='weighted'))

# 5. Создаем DataFrame как у тебя
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
df['target'] = y

print("\nDataFrame:")
print(df.head())
print(f"\nРазмеры: X {X.shape}, y {y.shape}")

# 6. Дополнительная визуализация - распределение по классам
plt.figure(figsize=(15, 5))

# Распределение признака 1 по классам
plt.subplot(1, 2, 1)
for class_label in np.unique(y):
    plt.hist(X[y == class_label, 0], alpha=0.6, label=f'Class {class_label}')
plt.title('Распределение Признака 1 по классам')
plt.xlabel('Значение Признака 1')
plt.ylabel('Частота')
plt.legend()

# Распределение признака 2 по классам
plt.subplot(1, 2, 2)
for class_label in np.unique(y):
    plt.hist(X[y == class_label, 1], alpha=0.6, label=f'Class {class_label}')
plt.title('Распределение Признака 2 по классам')
plt.xlabel('Значение Признака 2')
plt.ylabel('Частота')
plt.legend()

plt.tight_layout()
plt.show()

# 7. Тепловая карта корреляции
df_numeric = df[['Feature_1', 'Feature_2', 'target']]
plt.figure(figsize=(8, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Корреляционная матрица')
plt.show()
