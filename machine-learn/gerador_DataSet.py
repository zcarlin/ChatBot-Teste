import numpy as np
import pandas as pd

# Semente para reprodutibilidade
np.random.seed(42)

n_amostras_por_classe = 33334  # 3 x 33334 = 100002 amostras no total assim a gente consegue criar 100 mil dados

# Baixa Fertilidade
N_baixa = np.random.uniform(10, 40, n_amostras_por_classe)
P_baixa = np.random.uniform(5, 20, n_amostras_por_classe)
K_baixa = np.random.uniform(30, 70, n_amostras_por_classe)
pH_baixa = np.random.uniform(4.5, 6.0, n_amostras_por_classe)
MO_baixa = np.random.uniform(0.5, 2.0, n_amostras_por_classe)
Umidade_baixa = np.random.uniform(10, 30, n_amostras_por_classe)
classe_0 = np.zeros(n_amostras_por_classe, dtype=int)

# Media Fertilidade
N_media = np.random.uniform(25, 55, n_amostras_por_classe)
P_media = np.random.uniform(12, 28, n_amostras_por_classe)
K_media = np.random.uniform(50, 100, n_amostras_por_classe)
pH_media = np.random.uniform(5.5, 6.8, n_amostras_por_classe)
MO_media = np.random.uniform(1.2, 3.5, n_amostras_por_classe)
Umidade_media = np.random.uniform(15, 35, n_amostras_por_classe)
classe_1 = np.ones(n_amostras_por_classe, dtype=int)

# Alta Fertilidade
N_alta = np.random.uniform(45, 80, n_amostras_por_classe)
P_alta = np.random.uniform(20, 40, n_amostras_por_classe)
K_alta = np.random.uniform(80, 150, n_amostras_por_classe)
pH_alta = np.random.uniform(6.0, 7.5, n_amostras_por_classe)
MO_alta = np.random.uniform(2.5, 5.0, n_amostras_por_classe)
Umidade_alta = np.random.uniform(25, 40, n_amostras_por_classe)
classe_2 = np.full(n_amostras_por_classe, 2, dtype=int)

# Junta tudo
N_total = np.concatenate([N_baixa, N_media, N_alta])
P_total = np.concatenate([P_baixa, P_media, P_alta])
K_total = np.concatenate([K_baixa, K_media, K_alta])
pH_total = np.concatenate([pH_baixa, pH_media, pH_alta])
MO_total = np.concatenate([MO_baixa, MO_media, MO_alta])
Umidade_total = np.concatenate([Umidade_baixa, Umidade_media, Umidade_alta])
classes_total = np.concatenate([classe_0, classe_1, classe_2])

dados_solo = pd.DataFrame({
    'Nitrogenio_N': N_total,
    'Fosforo_P': P_total,
    'Potassio_K': K_total,
    'pH': pH_total,
    'Materia_Organica_pct': MO_total,
    'Umidade_pct': Umidade_total,
    'Classe_Fertilidade': classes_total
})

# embaralha tudo aqui
dados_solo = dados_solo.sample(frac=1, random_state=42).reset_index(drop=True)

# aqui colocamos ruido em 1% dos dados
num_ruido = int(0.01 * len(dados_solo))
idx_ruido = np.random.choice(dados_solo.index, num_ruido, replace=False)
dados_solo.loc[idx_ruido, 'pH'] += np.random.uniform(-2.0, 2.0, size=num_ruido)

# 3% de falta de dados, pra parecer mais com um dataset real
percentual_nan = 0.03
n_nan = int(len(dados_solo) * percentual_nan)

for coluna in ['Nitrogenio_N', 'Fosforo_P', 'Potassio_K', 'pH', 'Materia_Organica_pct', 'Umidade_pct']:
    indices_nan = np.random.choice(dados_solo.index, n_nan, replace=False)
    dados_solo.loc[indices_nan, coluna] = np.nan

# para os dados que faltam, colocaremos a media aqui
dados_solo.fillna(dados_solo.mean(numeric_only=True), inplace=True)

print("Amostras iniciais:")
print(dados_solo.head())
print("\nInformações do DataFrame:")
dados_solo.info()

# Salva aqui lol
dados_solo.to_csv(r"C:\Users\eliel\Desktop\ChatBot-Teste\machine-learn\Solos.csv", sep=";", index=False)