import numpy as np
from keras.models import load_model
from sklearn.decomposition import PCA

INPUT_PATH = 'y_smp_test.npy'
OUTPUT_PATH = 'predictions.npy'
#Входной файл
input_data = np.load(INPUT_PATH)
input_data = input_data.reshape(-1, 600)

pca = PCA(n_components=30)
input_data = pca.fit_transform(input_data)

#Загрузка модели
model = load_model('model.h5')
mean, q10, q25, q50, q75, q90 = model.predict(input_data)
#Предсказания
predictions = np.stack([mean, q10, q25, q50, q75, q90], axis=-1)
np.save(OUTPUT_PATH, predictions)
