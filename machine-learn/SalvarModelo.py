from tensorflow.keras.models import load_model
import pickle
import os

def salvarModelo(modelo, historico=None):
    print("\nOlá, sou EWACE! Deseja salvar o modelo criado por: Ayron, Eliel, Carlos e Walter?")

    pasta_destino = r"C:\Users\eliel\Desktop\ChatBot-Teste\Modelos_Treinados"
    os.makedirs(pasta_destino, exist_ok=True)  # Garante que a pasta existe

    while True:
        print("\n--- MENU DE SALVAMENTO ---")
        print("1: Salvar todo o modelo")
        print("2: Salvar somente os pesos")
        print("3: Salvar histórico do treinamento")
        print("0: Sair")

        entrada = input("Digite 1, 2, 3 ou 0 para sair: ")

        if entrada.isdigit():
            opc = int(entrada)

            if opc == 1:
                caminho_modelo = os.path.join(pasta_destino, "modelo_fertilidade.keras")
                print("Salvando todo o modelo...")
                modelo.save(caminho_modelo)
                print(f"\nModelo salvo em: {caminho_modelo}")
                print("Para carregar:\nfrom tensorflow.keras.models import load_model\nmodelo = load_model(\"modelo_fertilidade.keras\")")

            elif opc == 2:
                caminho_pesos = os.path.join(pasta_destino, "pesos_fertilidade.weights.h5")
                print("Salvando apenas os pesos...")
                modelo.save_weights(caminho_pesos)
                print(f"Pesos salvos em: {caminho_pesos}")
                print("Para carregar: modelo.load_weights(\"pesos_fertilidade.weights.h5\")")

            elif opc == 3:
                if historico is None:
                    print("Erro: histórico de treinamento não foi fornecido.")
                else:
                    caminho_hist = os.path.join(pasta_destino, "historico_treinamento.pkl")
                    print("Salvando histórico de treinamento...")
                    with open(caminho_hist, "wb") as f:
                        pickle.dump(historico.history, f)
                    print(f"Histórico salvo em: {caminho_hist}")
                    print("Para carregar:\nwith open(\"historico_treinamento.pkl\", \"rb\") as f:\n    historico = pickle.load(f)")

            elif opc == 0:
                print("Encerrando salvamento.")
                break

            else:
                print("Opção inválida. Tente novamente.")

        else:
            print(" Entrada inválida! Digite apenas números válidos.")