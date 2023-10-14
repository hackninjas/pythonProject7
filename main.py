import os

import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.src.layers import Conv1D, LSTM, Dense, MaxPooling1D, Bidirectional, \
    BatchNormalization, Dropout, Flatten
from keras.src.optimizers import Nadam
from sklearn.model_selection import train_test_split, KFold

# Проверка доступности графического процессора
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('Графический процессор доступен:', physical_devices[0])
else:
    print('Графический процессор не найден.')
# Установка Metal в качестве аппаратного ускорителя

X_train = np.load('y_smp_train.npy')
y_train = np.load('pars_smp_train.npy')
X_test = np.load('y_smp_test.npy')
y_train = np.squeeze(y_train)

print(X_train.shape)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Создаем функцию потерь RMSE Loss
def rmse_loss(y_true, y_pred):
    # Рассчитываем среднеквадратичную ошибку для каждого параметра
    square_errors = tf.square(y_true - y_pred)
    # Вычисляем взвешенную сумму квадратных ошибок
    weighted_rmse = tf.reduce_sum(square_errors, axis=-1)  # Предполагается, что у вас есть несколько параметров
    # Нормируем метрику для получения Lмеан
    rmse_normalized = tf.exp(-tf.sqrt(weighted_rmse))
    return rmse_normalized


def score(y_true, y_prede):
    print('')


# Функция потерь для 10-го квантиля
def quantile_loss_10(y_true, y_pred):
    tau = 0.10  # Значение квантиля
    error = y_true - y_pred
    loss = tf.reduce_mean(tf.maximum(tau * error, (tau - 1) * error), axis=-1)
    return loss


# Функция потерь для 25-го квантиля
def quantile_loss_25(y_true, y_pred):
    tau = 0.25  # Значение квантиля
    error = y_true - y_pred
    loss = tf.reduce_mean(tf.maximum(tau * error, (tau - 1) * error), axis=-1)
    return loss


# Функция потерь для 50-го квантиля
def quantile_loss_50(y_true, y_pred):
    tau = 0.50  # Значение квантиля
    error = y_true - y_pred
    loss = tf.reduce_mean(tf.maximum(tau * error, (tau - 1) * error), axis=-1)
    return loss


# Функция потерь для 75-го квантиля
def quantile_loss_75(y_true, y_pred):
    tau = 0.75  # Значение квантиля
    error = y_true - y_pred
    loss = tf.reduce_mean(tf.maximum(tau * error, (tau - 1) * error), axis=-1)
    return loss


# Функция потерь для 90-го квантиля
def quantile_loss_90(y_true, y_pred):
    tau = 0.90  # Значение квантиля
    error = y_true - y_pred
    loss = tf.reduce_mean(tf.maximum(tau * error, (tau - 1) * error), axis=-1)
    return loss


def build_model(input_dim=50, n_features=3):
    inputs = Input(shape=(input_dim, n_features))
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D()(x)
    x = Conv1D(filters=64, kernel_size=4, activation='relu')(x)
    x = MaxPooling1D()(x)
    x = Bidirectional(LSTM(units=32, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(units=64))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    mean = Dense(15, activation='linear', name='mean')(x)
    q10 = Dense(15, activation='linear', name='q10')(x)
    q25 = Dense(15, activation='linear', name='q25')(x)
    q50 = Dense(15, activation='linear', name='q50')(x)
    q75 = Dense(15, activation='linear', name='q75')(x)
    q90 = Dense(15, activation='linear', name='q90')(x)
    outputs = [mean, q10, q25, q50, q75, q90]
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss={
        'mean': rmse_loss,
        'q10': quantile_loss_10,
        'q25': quantile_loss_25,
        'q50': quantile_loss_50,
        'q75': quantile_loss_75,
        'q90': quantile_loss_90,
    }, optimizer=Nadam(),
        metrics={
            'mean': 'accuracy',
            'q10': 'accuracy',
            'q25': 'accuracy',
            'q50': 'accuracy',
            'q75': 'accuracy',
            'q90': 'accuracy',
        })
    print(model.summary())
    return model


# Определите количество разбиений для кросс-валидации
n_splits = 2
kf = KFold(n_splits=n_splits)

mean_rmse_scores = []
quantile_rmse_scores = {'q10': [], 'q25': [], 'q50': [], 'q75': [], 'q90': []}

input_dim = X_train.shape[1]
n_feats = X_train.shape[2]
model = build_model(input_dim, n_feats)

with tf.device('/GPU:0'):
    # Разделите данные и обучите/оцените модель для каждого разбиения
    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]
        model.fit(X_train_fold, y_train_fold, epochs=2, batch_size=50, validation_data=(X_val_fold, y_val_fold))

        rmse_scores = model.evaluate(X_val_fold, y_val_fold, batch_size=50)

        mean_rmse_scores.append(rmse_scores[1])  # среднее RMSE
        quantile_rmse_scores['q10'].append(rmse_scores[2])  # RMSE для квантили 10
        quantile_rmse_scores['q25'].append(rmse_scores[3])  # RMSE для квантили 25
        quantile_rmse_scores['q50'].append(rmse_scores[4])  # RMSE для квантили 50
        quantile_rmse_scores['q75'].append(rmse_scores[5])  # RMSE для квантили 75
        quantile_rmse_scores['q90'].append(rmse_scores[6])  # RMSE для квантили 90

    mean, q10, q25, q50, q75, q90 = model.predict(X_test)
    predictions = np.stack([mean, q10, q25, q50, q75, q90], axis=-1)

print(f'shape:{predictions.shape}')

# Выведите средние значения метрик RMSE для всех разбиений
print("Mean RMSE (Mean):", np.mean(mean_rmse_scores))
print("Mean RMSE (Q10):", np.mean(quantile_rmse_scores['q10']))
print("Mean RMSE (Q25):", np.mean(quantile_rmse_scores['q25']))
print("Mean RMSE (Q50):", np.mean(quantile_rmse_scores['q50']))
print("Mean RMSE (Q75):", np.mean(quantile_rmse_scores['q75']))
print("Mean RMSE (Q90):", np.mean(quantile_rmse_scores['q90']))
np.save('sample_submit.npy', predictions)
