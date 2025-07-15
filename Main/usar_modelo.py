import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# Carregar modelo, scaler e dados de teste
modelo = load_model('../Modelos/modelo_fertilidade.keras')
scaler = joblib.load('../Modelos/scaler_fertilidade.save')
dados = np.load('../Modelos/dados_teste.npz', allow_pickle=True)
X_teste = dados['X_teste']
y_teste = dados['y_teste']
# Para gráficos de barras
X = dados['X']
y = dados['y']

# Funções de gráficos e menu
def menu_graficos():
    while True:
        print("\nEscolha o gráfico que deseja visualizar:")
        print("1 - Gráfico de Barras (Pilar) das Médias das Features")
        print("2 - Histórico de Precisão e Perda do Treinamento")
        print("3 - Matriz de Confusão")
        print("0 - Sair")
        opcao = input("Digite o número da opção: ")
        if opcao == '1':
            plot_barras()
        elif opcao == '2':
            plot_historico_treinamento()
        elif opcao == '3':
            plot_matriz_confusao()
        elif opcao == '0':
            print("Saindo do menu de gráficos.")
            break
        else:
            print("Opção inválida. Tente novamente.")

def plot_barras():
    # Para exibir as médias das features originais
    # X está normalizado, mas podemos mostrar as médias
    medias = X.mean(axis=0)
    colunas = [f'feature{i+1}' for i in range(X.shape[1])]
    plt.figure(figsize=(10,6))
    sns.barplot(x=colunas, y=medias, palette='viridis')
    plt.title('Média das Features (Barras/Pilar)')
    plt.ylabel('Média Normalizada')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_historico_treinamento():
    print("Este gráfico só está disponível no script de treino, pois depende do histórico salvo durante o treinamento.")

def plot_matriz_confusao():
    from sklearn.metrics import confusion_matrix
    y_pred_prob = modelo.predict(X_teste)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_teste_classes = np.argmax(y_teste, axis=1)
    mat_conf = confusion_matrix(y_teste_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat_conf, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Baixa Fertilidade.', 'Média Fertilidade.', 'Alta Fertilidade.'], 
                yticklabels=['Baixa Fertilidade.', 'Média Fertilidade.', 'Alta Fertilidade.'])
    plt.title('Matriz de Confusão')
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Prevista')
    plt.show()

# Exemplo de predição com novos dados
def prever_novo():
    print("\nDigite os valores das features para previsão (na ordem do seu CSV):")
    valores = []
    for i in range(X.shape[1]):
        valor = float(input(f"Feature {i+1}: "))
        valores.append(valor)
    novo = np.array(valores).reshape(1, -1)
    novo_norm = scaler.transform(novo)
    pred = modelo.predict(novo_norm)
    classe_predita = np.argmax(pred, axis=1)[0]
    print('Classe prevista:', classe_predita)

if __name__ == "__main__":
    while True:
        print("\n1 - Menu de Gráficos\n2 - Fazer uma previsão\n0 - Sair")
        op = input("Escolha: ")
        if op == '1':
            menu_graficos()
        elif op == '2':
            prever_novo()
        elif op == '0':
            break
        else:
            print("Opção inválida.")