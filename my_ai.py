import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


class CodeClassifier:
    def __init__(self, model_type='nn'):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=None,
            token_pattern=r'(?u)\b\w+\b',
            analyzer='word'
        )
        self.model = None

    def _build_nn_model(self, input_dim):
        """Создание модели нейронной сети с правильным Input слоем"""
        model = Sequential([
            Input(shape=(input_dim,)),  # Правильный способ указания input_shape
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def preprocess_data(self, texts, labels=None, fit_vectorizer=False):
        if fit_vectorizer:
            X = self.vectorizer.fit_transform(texts)
        else:
            X = self.vectorizer.transform(texts)

        if labels is not None:
            y = np.array(labels)
            return X, y
        return X

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=500, batch_size=32):
        if self.model_type == 'nn':
            X_train_dense = X_train.toarray()
            input_dim = X_train_dense.shape[1]
            self.model = self._build_nn_model(input_dim)

            if X_val is not None:
                X_val_dense = X_val.toarray()
                self.model.fit(
                    X_train_dense, y_train,
                    validation_data=(X_val_dense, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1
                )
            else:
                self.model.fit(
                    X_train_dense, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1
                )
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

    def predict(self, X):
        """Предсказание меток для новых данных"""
        if self.model_type == 'nn':
            X_dense = X.toarray()
            predictions = self.model.predict(X_dense)
            return (predictions > 0.5).astype(int).flatten()
        else:
            return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Оценка качества модели на тестовых данных"""
        if self.model_type == 'nn':
            X_test_dense = X_test.toarray()
            loss, accuracy = self.model.evaluate(X_test_dense, y_test, verbose=0)
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Loss: {loss:.4f}")
            y_pred = self.predict(X_test)
        else:
            y_pred = self.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Test Accuracy: {accuracy:.4f}")

        # Добавляем проверку на наличие обоих классов
        if len(np.unique(y_test)) > 1 and len(np.unique(y_pred)) > 1:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Original', 'Copied'], zero_division=0))

            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
        else:
            print("\nWarning: Not enough classes in predictions for full evaluation")

    def save_model(self, model_path='code_classifier'):
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if self.model_type == 'nn':
            self.model.save(os.path.join(model_path, 'model.h5'))
        else:
            with open(os.path.join(model_path, 'rf_model.pkl'), 'wb') as f:
                pickle.dump(self.model, f)

        with open(os.path.join(model_path, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)

    @classmethod
    def load_model(cls, model_path='code_classifier', model_type='nn'):
        classifier = cls(model_type=model_type)

        with open(os.path.join(model_path, 'vectorizer.pkl'), 'rb') as f:
            classifier.vectorizer = pickle.load(f)

        if model_type == 'nn':
            classifier.model = tf.keras.models.load_model(os.path.join(model_path, 'model.h5'))
        else:
            with open(os.path.join(model_path, 'rf_model.pkl'), 'rb') as f:
                classifier.model = pickle.load(f)

        return classifier


def check(s):
    # 5. Пример предсказания
    new_codes = [s]

    X_new_vec = classifier.preprocess_data(new_codes)
    predictions = classifier.predict(X_new_vec)

    print("\nPredictions for new codes:")
    for code, pred in zip(new_codes, predictions):
        print(f"\nCode:\n{code}\nPrediction: {'Copied' if pred == 1 else 'Original'}")


# Пример использования с обработкой возможных ошибок
if __name__ == "__main__":
    try:
        # 1. Загрузка данных
        data = pd.read_csv('code_dataset.csv', sep='~')
        texts = data['code'].values
        labels = data['label'].values
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating dummy data for demonstration...")
        texts = [
            "def add(a, b):\n    return a + b",
            "function sum(x, y) {\n  return x + y;\n}",
            "def calculate_sum(first_number, second_number):\n    result = first_number + second_number\n    return result",
            "const square = num => num * num;",
            "def compute_total(initial_value, increment):\n    total = initial_value + increment\n    return total",
            "print('Hello, world!')"
        ]
        labels = np.array([0, 0, 1, 0, 1, 0])

    # 2. Разделение данных
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # 3. Инициализация и обучение
        classifier = CodeClassifier(model_type='nn')

        # Преобразование текста в признаки
        X_train_vec, y_train_vec = classifier.preprocess_data(X_train, y_train, fit_vectorizer=True)

        # Для нейронной сети можно добавить валидационный набор
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=0.5, random_state=42
        )
        X_val_vec, y_val_vec = classifier.preprocess_data(X_val, y_val)

        # Обучение модели
        classifier.train(X_train_vec, y_train_vec, X_val_vec, y_val_vec, epochs=250)

        # 4. Оценка качества
        X_test_vec, y_test_vec = classifier.preprocess_data(X_test, y_test)
        classifier.evaluate(X_test_vec, y_test_vec)

        # 6. Сохранение модели
        classifier.save_model()

    except Exception as e:
        print(f"Error during training/evaluation: {e}")
