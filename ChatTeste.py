import pandas as pd
import os
import unicodedata
import string
from sentence_transformers import SentenceTransformer, util

# ===============================
# Função de pré-processamento
def preprocessar_texto(texto):
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    texto = ' '.join(texto.split())
    return texto

# ===============================
# Carregar e preparar os dados
def carregar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv, sep=';', encoding='utf-8')
    df['input_text'] = df['input_text'].astype(str).apply(preprocessar_texto)
    return df

# ===============================
# Iniciar chatbot semântico
def iniciar_chat_semantico(df):
    print("\n🌱 Chatbot de Fertilidade do Solo (semântico)")
    print("Digite uma frase sobre seu solo. Ex: 'meu solo está fraco e seco'")
    print("Digite 'sair' para encerrar.\n")

    modelo_st = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    perguntas = df['input_text'].tolist()
    respostas = df['resposta'].tolist()

    print("🧠 Gerando embeddings do dataset...")
    embeddings_perguntas = modelo_st.encode(perguntas, convert_to_tensor=True)

    while True:
        entrada = input("Você: ").strip()
        if entrada.lower() == 'sair':
            print("👋 Até logo!")
            break

        entrada_proc = preprocessar_texto(entrada)
        embedding_usuario = modelo_st.encode(entrada_proc, convert_to_tensor=True)

        similaridades = util.cos_sim(embedding_usuario, embeddings_perguntas)
        indice_mais_proximo = int(similaridades.argmax())
        confianca = float(similaridades.max())

        if confianca < 0.65:
            print("Bot: Desculpe, não entendi sua pergunta. Pode reformular?")
        else:
            resposta = respostas[indice_mais_proximo]
            print("Bot:", resposta)

# ===============================
# Execução principal
if __name__ == "__main__":
    caminho_dados = r"C:\Users\eliel\Desktop\ChatBot-Teste\Dados\dataset_expandido_balanceado.csv"
    
    if not os.path.exists(caminho_dados):
        print(f"❌ Dataset não encontrado: {caminho_dados}")
    else:
        base = carregar_dados(caminho_dados)
        iniciar_chat_semantico(base)
