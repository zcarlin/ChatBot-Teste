import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os

# Carregar dados
caminho_csv = r'C:\Users\zCarlin\Desktop\ChatBot-Teste-main\ChatBot-Teste-main\Dados\Solos.csv'
dados_solo = pd.read_csv(caminho_csv, sep=';')
X = dados_solo.drop('Classe_Fertilidade', axis=1)
y = dados_solo['Classe_Fertilidade']

# Normalizar e codificar
scaler = MinMaxScaler()
X_normalizado = scaler.fit_transform(X)
y_categorico = to_categorical(y)

# Dividir dados
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_normalizado, y_categorico, test_size=0.2, random_state=42, stratify=y)

# Definir modelo
modelo_solo = Sequential([
    Dense(32, input_shape=(X_treino.shape[1],), activation='relu'),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(y_treino.shape[1], activation='softmax')
])

modelo_solo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Treinar
historico_treinamento = modelo_solo.fit(X_treino, y_treino, epochs=60, batch_size=16,
                                        validation_data=(X_teste, y_teste),
                                        verbose=1,
                                        callbacks=[early_stop, reduce_lr])

# Salvar modelo, scaler e dados de teste na pasta Modelos
PASTA_MODELOS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Modelos'))
os.makedirs(PASTA_MODELOS, exist_ok=True)
modelo_solo.save(os.path.join(PASTA_MODELOS, 'modelo_fertilidade.keras'))
joblib.dump(scaler, os.path.join(PASTA_MODELOS, 'scaler_fertilidade.save'))
np.savez_compressed(os.path.join(PASTA_MODELOS, 'dados_teste.npz'), X_teste=X_treino, y_teste=y_treino, X=X, y=y)
print('Modelo, scaler e dados de teste salvos com sucesso na pasta Modelos!')