import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.src.layers import Conv1D, LSTM, Dense, MaxPooling1D, Bidirectional, \
    BatchNormalization, Dropout, Flatten
from sklearn.model_selection import train_test_split

# Проверка доступности графического процессора
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('Графический процессор доступен:', physical_devices[0])
else:
    print('Графический процессор не найден.')


# Установка Metal в качестве аппаратного ускорителя
def normalize_data(matrix):
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    normalized_matrix = (matrix - mean) / std
    return normalized_matrix


X_train = np.load('y_smp_train.npy')
y_train = np.load('pars_smp_train.npy')
X_test = np.load('y_smp_test.npy')
y_train = np.squeeze(y_train)

X_train = normalize_data(X_train)
X_test = normalize_data(X_test)
y_train = normalize_data(y_train)

print(X_train.shape)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Создаем функцию потерь RMSE Loss
def wrmse_loss(y_true, y_pred):
    # Расчет квадратов ошибок
    square_errors = tf.square(y_true - y_pred)
    # Вычисление взвешенной суммы квадратных ошибок
    weighted_rmse = tf.reduce_sum(tf.minimum(square_errors, 0.01 * tf.ones_like(square_errors)), axis=-1)
    # Нормализация метрики для получения WRMSE
    wrmse_normalized = tf.exp(-tf.sqrt(weighted_rmse))
    return wrmse_normalized


def calculate_score(y_true, y_pred):
    # Веса для каждой части метрики
    w_rmse = 0.45
    w_quantiles = 0.45
    w_calibration = 0.1

    # Расчет точности предсказания среднего
    rmse = np.sqrt(np.mean(np.minimum((y_pred - y_true) ** 2, 0.01)))
    score_rmse = np.exp(-rmse)

    # Расчет точности предсказания квантилей
    quantiles = np.percentile(np.abs(y_pred - y_true), [10, 25, 50, 75, 90])
    score_quantiles = np.exp(-(0.002 * np.sum(quantiles)))

    # Расчет проверки калибровки
    calibration = np.abs(np.mean(np.less_equal(y_pred, y_true)) - 0.5)
    score_calibration = np.exp(-(0.25 * calibration))

    # Итоговый скор
    final_score = w_rmse * score_rmse + w_quantiles * score_quantiles + w_calibration * score_calibration

    return final_score


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
        'mean': wrmse_loss,
        'q10': quantile_loss_10,
        'q25': quantile_loss_25,
        'q50': quantile_loss_50,
        'q75': quantile_loss_75,
        'q90': quantile_loss_90,
    }, optimizer='adam')
    print(model.summary())
    return model


mean_rmse_scores = []
quantile_rmse_scores = {'q10': [], 'q25': [], 'q50': [], 'q75': [], 'q90': []}

input_dim = X_train.shape[1]
n_feats = X_train.shape[2]
model = build_model(input_dim, n_feats)

with tf.device('/GPU:0'):
    model.fit(X_train, y_train, epochs=2, batch_size=256, validation_split=0.2, verbose=2)
    rmse_scores = model.evaluate(X_val, y_val, batch_size=256)
    print(f'mean :{calculate_score(y_val, rmse_scores[1])}')
    mean_rmse_scores.append(rmse_scores[1])  # среднее RMSE
    quantile_rmse_scores['q10'].append(rmse_scores[2])  # RMSE для квантили 10
    print(f'10 :{calculate_score(y_val, rmse_scores[2])}')
    quantile_rmse_scores['q25'].append(rmse_scores[3])  # RMSE для квантили 25
    print(f'25 :{calculate_score(y_val, rmse_scores[3])}')
    quantile_rmse_scores['q50'].append(rmse_scores[4])  # RMSE для квантили 50
    print(f'50 :{calculate_score(y_val, rmse_scores[4])}')
    quantile_rmse_scores['q75'].append(rmse_scores[5])  # RMSE для квантили 75
    print(f'75 :{calculate_score(y_val, rmse_scores[5])}')
    quantile_rmse_scores['q90'].append(rmse_scores[6])  # RMSE для квантили 90
    print(f'90 :{calculate_score(y_val, rmse_scores[6])}')

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
