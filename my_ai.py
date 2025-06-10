import Levenshtein
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
            Input(shape=(input_dim,)),
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
            x = self.vectorizer.fit_transform(texts)
        else:
            x = self.vectorizer.transform(texts)

        if labels is not None:
            y = np.array(labels)
            return x, y
        return x

    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=250, batch_size=32):
        if self.model_type == 'nn':
            x_train_dense = x_train.toarray()
            input_dim = x_train_dense.shape[1]
            self.model = self._build_nn_model(input_dim)

            if x_val is not None:
                x_val_dense = x_val.toarray()
                self.model.fit(
                    x_train_dense, y_train,
                    validation_data=(x_val_dense, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1
                )
            else:
                self.model.fit(
                    x_train_dense, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1
                )
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(x_train, y_train)

    def predict(self, x):
        """Предсказание меток для новых данных"""
        if self.model_type == 'nn':
            x_dense = x.toarray()
            predictions = self.model.predict(x_dense)
            return predictions
        else:
            return self.model.predict(x)

    def evaluate(self, x_test, y_test):
        """Оценка качества модели на тестовых данных"""
        if self.model_type == 'nn':
            x_test_dense = x_test.toarray()
            loss, accuracy = self.model.evaluate(x_test_dense, y_test, verbose=0)
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Loss: {loss:.4f}")
            y_pred = self.predict(x_test)
        else:
            y_pred = self.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Test Accuracy: {accuracy:.4f}")

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


def create_model(data_path='code_dataset.csv', model_type='nn', model_save_path='code_classifier',
                           epochs=500):
    try:
        # Чтение CSV с обработкой запятых в коде
        data = pd.read_csv(data_path, sep='~')

        # Проверка структуры данных
        if len(data.columns) != 2:
            raise ValueError("CSV должен содержать ровно 2 столбца: code и label")

        data.columns = ['code', 'label']  # Принудительно задаем названия столбцов
        texts = data['code'].values
        labels = data['label'].values

    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        print("Используем демонстрационные данные...")
        texts = [
            "def add(a, b): return a + b",
            "print('Hello, world!')",
            "def func(x, y):\n    return x * y",
            "def calculate(a, b):\n    result = a + b\n    return result",
            "console.log('test')"
        ]
        labels = np.array([0, 0, 1, 1, 0])

    # Разделение данных
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Создание и обучение классификатора
    classifier = CodeClassifier(model_type=model_type)

    # Преобразование текста в признаки
    x_train_vec, y_train_vec = classifier.preprocess_data(x_train, y_train, fit_vectorizer=True)

    # Для нейросети добавляем валидационный набор
    if model_type == 'nn':
        x_val, x_test, y_val, y_test = train_test_split(
            x_test, y_test, test_size=0.5, random_state=42
        )
        x_val_vec, y_val_vec = classifier.preprocess_data(x_val, y_val)
        classifier.train(x_train_vec, y_train_vec, x_val_vec, y_val_vec, epochs=epochs)
    else:
        classifier.train(x_train_vec, y_train_vec)

    # Сохранение модели
    classifier.save_model(model_save_path)

    return classifier


def load_model(model_path='code_classifier', model_type='nn'):
    try:
        classifier = CodeClassifier.load_model(model_path=model_path, model_type=model_type)

        print("Модель успешно загружена. Готов к классификации.")
        return classifier

    except Exception as e:
        print(f"Ошибка при загрузке модели или классификации: {e}")
        raise


def check(classifier, code_sample):
    # Преобразование входных данных
    x_vec = classifier.preprocess_data([code_sample])

    # Выполнение предсказания
    predictions = classifier.predict(x_vec)

    return round(predictions[0][0] * 100.0, 4)


def compare(code1, code2):
    def normalize_code(code):
        lines = [line.strip() for line in code.splitlines() if line.strip()]
        return '\n'.join(lines)

    norm_code1 = normalize_code(code1)
    norm_code2 = normalize_code(code2)

    # Сходство на основе строк (Левенштейн)
    text_similarity = Levenshtein.ratio(norm_code1, norm_code2) * 100

    return round(text_similarity, 2)
