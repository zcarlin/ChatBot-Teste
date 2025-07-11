import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
import unicodedata
import string
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from difflib import SequenceMatcher

# Função de pré-processamento robusto
def preprocessar_texto(texto):
    # Minúsculas
    texto = texto.lower()
    # Remover acentuação
    texto = ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    # Remover pontuação
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    # Remover espaços extras
    texto = ' '.join(texto.split())
    return texto

# ==== PARTE 1 - CARREGAMENTO E TREINAMENTO ====
def treinar_chatbot(caminho_csv):
    df = pd.read_csv(caminho_csv, sep=';', encoding='utf-8')
    # Pré-processar textos
    df['input_text'] = df['input_text'].astype(str).apply(preprocessar_texto)

    # Tokenização
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(df['input_text'])
    sequences = tokenizer.texts_to_sequences(df['input_text'])
    padded = pad_sequences(sequences, padding='post')

    # Encode dos rótulos
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['intent'])

    # Modelo NLP otimizado
    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=padded.shape[1]),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(len(set(labels)), activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        padded, np.array(labels),
        epochs=100,
        verbose=1,
        validation_split=0.1,  # 10% dos dados para validação
        callbacks=[early_stop]
    )

    stopped_epoch = early_stop.stopped_epoch
    if stopped_epoch == 0:
        stopped_epoch = len(history.history['loss']) - 1

    best_val_loss = min(history.history['val_loss'])
    best_epoch = history.history['val_loss'].index(best_val_loss) + 1

    print(f"\n✅ Treinamento interrompido na época: {stopped_epoch + 1}")
    print(f"⭐ Melhor época: {best_epoch}")
    print(f"📉 Melhor val_loss: {best_val_loss:.4f}\n")

    return model, tokenizer, label_encoder, df

# Função para encontrar resposta mais similar

def resposta_mais_proxima(entrada, respostas):
    def similaridade(a, b):
        return SequenceMatcher(None, a, b).ratio()
    return max(respostas, key=lambda r: similaridade(entrada, r))

# ==== PARTE 2 - CHAT INTERATIVO ====
def iniciar_chat(model, tokenizer, label_encoder, df):
    print("\n🌿 Chatbot de Fertilidade do Solo")
    print("Digite uma frase sobre seu solo. Ex: 'meu solo está fraco e seco'")
    print("Digite 'sair' para encerrar.\n")

    ultima_resposta_conceitual = None
    padroes_conceituais = [
        'o que e', 'explique', 'para que serve', 'o que significa', 'me explique'
    ]
    padroes_genericos = [
        'o que e isso', 'mas o que e isso', 'mas o que significa isso'
    ]

    while True:
        entrada = input("Você: ").strip()
        entrada_proc = preprocessar_texto(entrada)
        if entrada_proc == 'sair':
            print("👋 Até logo!")
            break

        # Se a pergunta for genérica e há contexto, repete a última explicação conceitual
        if any(p in entrada_proc for p in padroes_genericos) and ultima_resposta_conceitual:
            print("Bot:", ultima_resposta_conceitual)
            continue

        seq = tokenizer.texts_to_sequences([entrada_proc])
        padded_seq = pad_sequences(seq, maxlen=model.input_shape[1], padding='post')

        pred = model.predict(padded_seq, verbose=0)
        confianca = np.max(pred)
        intent_index = np.argmax(pred)

        if confianca < 0.7:
            print("Bot: Desculpe, não entendi sua pergunta. Pode reformular?")
            continue

        intent_predita = label_encoder.inverse_transform([intent_index])[0]
        respostas = df[df['intent'] == intent_predita]['resposta'].tolist()
        perguntas = df[df['intent'] == intent_predita]['input_text'].tolist()

        if respostas and perguntas:
            # Escolher a resposta associada à pergunta mais similar ao input
            def similaridade(a, b):
                from difflib import SequenceMatcher
                return SequenceMatcher(None, a, b).ratio()
            idx_mais_similar = max(range(len(perguntas)), key=lambda i: similaridade(entrada_proc, perguntas[i]))
            resposta = respostas[idx_mais_similar]
            print("Bot:", resposta)
            # Se a pergunta for conceitual, salva resposta para contexto
            if any(p in entrada_proc for p in padroes_conceituais):
                ultima_resposta_conceitual = resposta
            else:
                ultima_resposta_conceitual = None
        else:
            print("Bot: Não tenho uma resposta para isso no momento.")

# ==== EXECUÇÃO PRINCIPAL ====
if __name__ == "__main__":
    caminho_dados = r"C:\Users\aluno vespertino\Desktop\Carlos\Carlos\Dados\dataset_expandido_balanceado.csv"

    if not os.path.exists(caminho_dados):
        print(f"❌ Dataset não encontrado: {caminho_dados}")
    else:
        modelo, tokenizer, encoder, base = treinar_chatbot(caminho_dados)
        iniciar_chat(modelo, tokenizer, encoder, base)
