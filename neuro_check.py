import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 1. Пример данных (в реальности нужно больше данных!)
texts = [
    "def solve(x): return x * 2",  # Оригинальное решение
    "def solve(x): return x + x",  # Похоже на списанное
    "def calculate(y): return y * 2",  # Оригинальное, но другая структура
    "def solve(a): return a * 2",  # Почти копия (списано)
    "def sl(m) return m * 5",
    "def to_lower(string) return string.lower()"
]
labels = np.array([0, 1, 0, 1, 0, 1])  # 0 = не списано, 1 = списано

# 2. Преобразуем текст в числовые признаки (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts).toarray()

# 3. Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 4. Создаем нейросеть
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Обучаем модель
model.fit(X_train, y_train, epochs=200, batch_size=2, validation_split=0.1)

# 6. Проверяем точность
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nТочность модели: {accuracy * 100:.2f}%")

# 7. ПРЕДСКАЗАНИЕ НА НОВОМ ТЕКСТЕ
def predict_plagiarism(code):
    # Векторизуем входной текст
    vec = vectorizer.transform([code]).toarray()
    # Предсказываем вероятность
    prob = model.predict(vec)[0][0]
    return prob * 100
