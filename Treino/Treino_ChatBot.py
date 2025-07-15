import pandas as pd
import os
import unicodedata
import string
import joblib
import json
import datetime
import time
import glob
import re
import torch
import numpy as np
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
# Salvar modelo e dados como .pkl
def salvar_modelo_e_dados(modelo_st, embeddings, df, caminho='modelo_semantico.pkl'):
    print("💾 Salvando modelo e dados em .pkl...")
    
    joblib.dump({
        'modelo': modelo_st,
        'embeddings': embeddings.cpu().numpy(),  # Convertido para numpy (mais leve)
        'respostas': df['resposta'].tolist(),
        'perguntas': df['input_text'].tolist()
    }, caminho)

    print(f"✅ Modelo e dados salvos em: {caminho}")

# ===============================
# Carregar modelo e dados .pkl
def carregar_modelo_e_dados(caminho='modelo_semantico.pkl'):
    print("📂 Carregando modelo e dados de .pkl...")
    
    dados = joblib.load(caminho)
    modelo_st = dados['modelo']
    embeddings = torch.tensor(dados['embeddings'])  # Reconvertido para tensor
    perguntas = dados['perguntas']
    respostas = dados['respostas']

    return modelo_st, embeddings, perguntas, respostas

# ===============================
# Memória contextual
def extrair_contexto(entrada):
    contexto = {
        'tipo_solo': None,
        'nivel_fertilidade': None,
        'problema': None,
        'acao_desejada': None
    }
    entrada_lower = entrada.lower()
    
    tipos_solo = ['arenoso', 'argiloso', 'humoso', 'calcário', 'ácido', 'alcalino']
    for tipo in tipos_solo:
        if tipo in entrada_lower:
            contexto['tipo_solo'] = tipo
            break
    
    niveis = ['alta', 'média', 'baixa', 'alta fertilidade', 'média fertilidade', 'baixa fertilidade']
    for nivel in niveis:
        if nivel in entrada_lower:
            contexto['nivel_fertilidade'] = nivel
            break
    
    problemas = ['seco', 'úmido', 'compactado', 'fraco', 'pobre', 'ácido', 'alcalino']
    for problema in problemas:
        if problema in entrada_lower:
            contexto['problema'] = problema
            break
    
    acoes = ['melhorar', 'adubar', 'nutrir', 'fortalecer', 'recuperar', 'corrigir']
    for acao in acoes:
        if acao in entrada_lower:
            contexto['acao_desejada'] = acao
            break
    
    return contexto

def atualizar_contexto(contexto_atual, nova_entrada):
    novo_contexto = extrair_contexto(nova_entrada)
    contexto_final = contexto_atual.copy() if contexto_atual else {}
    for k, v in novo_contexto.items():
        if v:
            contexto_final[k] = v
    return contexto_final

def expandir_pergunta_com_contexto(entrada, contexto_anterior):
    if not contexto_anterior:
        return entrada
    
    if re.search(r'\b(ele|ela|isso|isso mesmo)\b', entrada.lower()):
        pergunta_expandida = entrada
        if contexto_anterior.get('tipo_solo'):
            pergunta_expandida = f"meu solo {contexto_anterior['tipo_solo']} {entrada}"
        elif contexto_anterior.get('nivel_fertilidade'):
            pergunta_expandida = f"meu solo com {contexto_anterior['nivel_fertilidade']} {entrada}"
        elif contexto_anterior.get('problema'):
            pergunta_expandida = f"meu solo {contexto_anterior['problema']} {entrada}"
        return pergunta_expandida
    return entrada

# ===============================
# Histórico de sessões
def criar_pasta_historico(pasta='historico'):
    os.makedirs(pasta, exist_ok=True)
    return pasta

def salvar_sessao(id_sessao, conversas, pasta='historico'):
    dados = {
        "id": id_sessao,
        "data": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "conversas": conversas
    }
    caminho_arquivo = os.path.join(pasta, f"{id_sessao}.json")
    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

def listar_sessoes(pasta='historico'):
    arquivos = glob.glob(os.path.join(pasta, "*.json"))
    sessoes = []
    for arq in arquivos:
        try:
            with open(arq, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                sessoes.append({'arquivo': arq, 'id': dados['id'], 'data': dados['data']})
        except:
            pass
    return sorted(sessoes, key=lambda x: x['data'], reverse=True)

def carregar_sessao(arquivo):
    with open(arquivo, 'r', encoding='utf-8') as f:
        return json.load(f)

def gerar_id_sessao():
    timestamp = int(time.time())
    return f"sessao_{timestamp}"

# ===============================
# Chatbot com memória contextual
def iniciar_chat_semantico(modelo_st, embeddings_perguntas, perguntas, respostas):
    print("\n🌱 Chatbot de Fertilidade do Solo (semântico e com contexto)")
    print("Digite uma frase sobre seu solo. Ex: 'meu solo está fraco e seco'")
    print("Digite 'sair' para encerrar.\n")
    
    criar_pasta_historico()
    
    sessoes = listar_sessoes()
    conversas = []
    contexto_atual = None
    
    if sessoes:
        resposta = input("Deseja continuar uma sessão anterior? (sim/não): ").strip().lower()
        if resposta in ['sim', 's', 'yes', 'y']:
            print("\nSessões disponíveis:")
            for i, sessao in enumerate(sessoes):
                print(f"[{i+1}] {sessao['data']} - {sessao['id']}")
            try:
                escolha = int(input("Escolha o número da sessão: "))
                if 1 <= escolha <= len(sessoes):
                    dados_sessao = carregar_sessao(sessoes[escolha-1]['arquivo'])
                    conversas = dados_sessao['conversas']
                    print(f"\n📜 Histórico da sessão {dados_sessao['id']} ({dados_sessao['data']}):")
                    for c in conversas:
                        print(f"Você: {c['entrada']}")
                        print(f"Bot: {c['resposta']}")
                        print("-"*30)
                    if conversas:
                        contexto_atual = conversas[-1].get('contexto')
            except:
                print("Entrada inválida. Começando nova sessão.")
                conversas = []
    
    id_sessao = gerar_id_sessao()
    
    while True:
        entrada = input("Você: ").strip()
        if entrada.lower() == 'sair':
            break
        
        contexto_atual = atualizar_contexto(contexto_atual, entrada)
        entrada_exp = expandir_pergunta_com_contexto(entrada, contexto_atual)
        
        entrada_proc = preprocessar_texto(entrada_exp)
        embedding_usuario = modelo_st.encode(entrada_proc, convert_to_tensor=True)
        
        similaridades = util.cos_sim(embedding_usuario, embeddings_perguntas)
        indice_mais_proximo = int(similaridades.argmax())
        confianca = float(similaridades.max())
        
        if confianca < 0.65:
            resposta_bot = "Desculpe, não entendi sua pergunta. Pode reformular?"
        else:
            resposta_bot = respostas[indice_mais_proximo]
        
        print(f"Bot: {resposta_bot}")
        print(f"Confiança: {confianca:.2f}\n")
        
        conversas.append({
            'entrada': entrada,
            'resposta': resposta_bot,
            'confianca': confianca,
            'contexto': contexto_atual
        })
    
    salvar_sessao(id_sessao, conversas)
    print(f"\n💾 Histórico salvo na sessão: {id_sessao}")
    print("👋 Até logo!")

# ===============================
# Execução principal
if __name__ == "__main__":
    caminho_dados = r"C:\Users\zCarlin\Desktop\ChatBot-Teste-main\ChatBot-Teste-main\Dados\dataset_expandido_balanceado.csv"
    caminho_modelo = "modelo_semantico.pkl"

    if not os.path.exists(caminho_dados):
        print(f"❌ Dataset não encontrado: {caminho_dados}")
    else:
        if os.path.exists(caminho_modelo):
            modelo_st, embeddings, perguntas, respostas = carregar_modelo_e_dados(caminho_modelo)
        else:
            df = carregar_dados(caminho_dados)
            modelo_st = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            print("🧠 Gerando embeddings...")
            embeddings = modelo_st.encode(df['input_text'].tolist(), convert_to_tensor=True)
            salvar_modelo_e_dados(modelo_st, embeddings, df, caminho_modelo)
            perguntas = df['input_text'].tolist()
            respostas = df['resposta'].tolist()

        iniciar_chat_semantico(modelo_st, embeddings, perguntas, respostas)