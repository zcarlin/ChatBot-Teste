import sys
import os
import unicodedata
import string
import joblib
import json
import datetime
import glob
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QColor, QTextCursor, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QLabel,
    QDialog,
    QDialogButtonBox,
    QGraphicsOpacityEffect,
)
from sentence_transformers import util

# --- PASTA PARA SALVAR HISTÓRICO ---
PASTA_HISTORICO = "historico"
os.makedirs(PASTA_HISTORICO, exist_ok=True)

# ==========================================================
# Função de pré-processamento
# ==========================================================

def preprocessar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )
    texto = texto.translate(str.maketrans("", "", string.punctuation))
    texto = " ".join(texto.split())
    return texto

# ==========================================================
# Chatbot adaptado para carregar de .pkl
# ==========================================================

class Chatbot:
    def __init__(self, caminho_modelo_pkl: str):
        print("📂 Carregando modelo e dados do arquivo .pkl...")
        dados = joblib.load(caminho_modelo_pkl)
        self.modelo_st = dados['modelo']
        self.embeddings_perguntas = dados['embeddings']
        self.perguntas = dados['perguntas']
        self.respostas = dados['respostas']

    def get_response(self, entrada_usuario: str) -> str:
        entrada_proc = preprocessar_texto(entrada_usuario)
        embedding_usuario = self.modelo_st.encode(entrada_proc, convert_to_tensor=True)

        similaridades = util.cos_sim(embedding_usuario, self.embeddings_perguntas)
        indice_mais_proximo = int(similaridades.argmax())
        confianca = float(similaridades.max())

        if confianca < 0.65:
            return "Desculpe, não entendi sua pergunta. Pode reformular?"
        return self.respostas[indice_mais_proximo]

# ==========================================================
# Sistema de histórico: salvar, carregar e deletar sessões
# ==========================================================

def deletar_sessao(caminho_arquivo):
    try:
        os.remove(caminho_arquivo)
        print(f"🗑️ Sessão removida: {caminho_arquivo}")
        return True
    except OSError as e:
        print(f"❌ Erro ao remover sessão: {e}")
        return False

def listar_sessoes():
    arquivos = glob.glob(os.path.join(PASTA_HISTORICO, "*.json"))
    sessoes = []
    for arquivo in arquivos:
        try:
            with open(arquivo, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                sessoes.append({
                    'arquivo': arquivo,
                    'id': dados['id'],
                    'data': dados['data'],
                    'conversas': dados['conversas']
                })
        except:
            continue
    return sorted(sessoes, key=lambda x: x['data'], reverse=True)

def salvar_sessao(id_sessao, conversas):
    dados = {
        "id": id_sessao,
        "data": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "conversas": conversas
    }
    nome_arquivo = f"{id_sessao}.json"
    caminho_arquivo = os.path.join(PASTA_HISTORICO, nome_arquivo)
    with open(caminho_arquivo, 'w', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    print(f"💾 Histórico salvo: {caminho_arquivo}")

def carregar_sessao(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==========================================================
# Dialogo para escolher sessão de histórico
# ==========================================================

class EscolherSessaoDialog(QDialog):
    def __init__(self, sessoes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Escolha uma sessão para carregar")
        self.setMinimumSize(400, 300)
        layout = QVBoxLayout(self)

        self.lista = QListWidget()
        for sessao in sessoes:
            item = QListWidgetItem(f"{sessao['data']} - {sessao['id']}")
            item.setData(Qt.ItemDataRole.UserRole, sessao['arquivo'])
            self.lista.addItem(item)
        layout.addWidget(QLabel("Selecione uma sessão para continuar:"))
        layout.addWidget(self.lista)

        botoes = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        botoes.accepted.connect(self.accept)
        botoes.rejected.connect(self.reject)
        layout.addWidget(botoes)

    def get_selecao(self):
        item = self.lista.currentItem()
        if item:
            return item.data(Qt.ItemDataRole.UserRole)
        return None

# ==========================================================
# Widget customizado para item de sessão com botão de deletar
# ==========================================================
class SessaoItemWidget(QWidget):
    def __init__(self, sessao, delete_callback, parent=None):
        super().__init__(parent)
        self.sessao = sessao
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        info_label = QLabel(f"🗓️ {sessao['data']}\nID: {sessao['id']}")
        info_label.setWordWrap(True)

        delete_button = QPushButton("🗑️")
        delete_button.setFixedSize(30, 30)
        delete_button.setCursor(Qt.CursorShape.PointingHandCursor)
        delete_button.setStyleSheet("""
            QPushButton { 
                background-color: transparent; 
                border: none; 
                font-size: 16px; 
                color: #AABBCB;
            }
            QPushButton:hover { 
                color: #E8EAED;
                background-color: #DD2C00;
                border-radius: 15px;
            }
        """)
        delete_button.clicked.connect(lambda: delete_callback(sessao['arquivo']))
        
        layout.addWidget(info_label)
        layout.addStretch()
        layout.addWidget(delete_button)

# ==========================================================
# Interface gráfica (GUI) — ChatbotWindow
# ==========================================================

class ChatbotWindow(QMainWindow):
    def __init__(self, chatbot: Chatbot):
        super().__init__()
        self.setWindowTitle("CHAT-GPT13")
        self.setGeometry(100, 100, 1000, 650)

        self.chatbot = chatbot
        self.conversas = []
        self.id_sessao = f"sessao_{int(datetime.datetime.now().timestamp())}"
        self.animating = False

        # --- Painel lateral para histórico ---
        self.sidebar = QListWidget()
        self.sidebar.setMaximumWidth(260)
        self.sidebar.setStyleSheet("""
            QListWidget {
                background-color: #181A20; 
                color: #E8EAED; 
                border-top-left-radius: 12px; 
                border-bottom-left-radius: 12px; 
                font-size: 14px;
                padding: 8px;
            }
            QListWidget::item:hover {
                background-color: #2A2D35;
            }
            QListWidget::item:selected {
                background-color: #4A90E2;
            }
        """)
        self.sidebar.itemClicked.connect(self.carregar_sessao_sidebar)
        

        # --- Título estilizado ---
        self.titulo = QLabel('<b>CHAT-GPT13</b>')
        self.titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.titulo.setStyleSheet("font-size: 28px; color: #4A90E2; padding: 18px 0 10px 0; letter-spacing: 2px;")

        # --- Área do chat ---
        self.chat_area = QTextEdit(self)
        self.chat_area.setReadOnly(True)
        font = QFont()
        font.setPointSize(15)
        self.chat_area.setFont(font)
        self.chat_area.setStyleSheet("background-color: #23272F; color: #E8EAED; border-radius: 16px; padding: 18px; margin: 0 0 8px 0;")
        self.opacity_effect = QGraphicsOpacityEffect(self.chat_area)
        self.chat_area.setGraphicsEffect(self.opacity_effect)

        # --- Campo de entrada ---
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Digite sua mensagem…")
        self.input_field.setFont(font)
        self.input_field.returnPressed.connect(self.send_message)
        self.input_field.setStyleSheet("background-color: #181A20; color: #E8EAED; padding: 14px; border-radius: 12px; font-size: 15px;")

        # --- Botão enviar ---
        self.send_button = QPushButton("Enviar", self)
        self.send_button.setFont(font)
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setStyleSheet("background-color: #4A90E2; color: white; padding: 12px 28px; border-radius: 12px; font-size: 15px;")

        # --- Layouts ---
        chat_layout = QVBoxLayout()
        chat_layout.addWidget(self.titulo)
        chat_layout.addWidget(self.chat_area)

        bottom = QHBoxLayout()
        bottom.addWidget(self.input_field)
        bottom.addWidget(self.send_button)
        chat_layout.addLayout(bottom)

        chat_container = QWidget()
        chat_container.setLayout(chat_layout)
        chat_container.setStyleSheet("background-color: #202123; border-top-right-radius: 16px; border-bottom-right-radius: 16px;")

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(chat_container)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.chat_area.append(
            f'<b><img src="resources/FazoL.png" width="36" height="36" style="border-radius: 18px; vertical-align: middle;"> Companheiro:</b> '
            "Olá! Sou seu assistente de fertilidade do solo. Como posso ajudar?<br>"
        )

        self.atualizar_sidebar()


    def atualizar_sidebar(self):
        self.sidebar.clear()

        new_chat_item = QListWidgetItem("➕ Nova Conversa")
        self.sidebar.addItem(new_chat_item)
        
        sessoes = listar_sessoes()
        for sessao in sessoes:
            item = QListWidgetItem(self.sidebar)
            widget = SessaoItemWidget(sessao, self.deletar_e_atualizar_sessao)
            item.setSizeHint(widget.sizeHint())
            item.setData(Qt.ItemDataRole.UserRole, sessao['arquivo']) 
            self.sidebar.setItemWidget(item, widget)

    def deletar_e_atualizar_sessao(self, caminho_arquivo):
        confirm = QMessageBox.question(self, "Confirmar Exclusão", 
                                      "Tem certeza que deseja excluir esta sessão?\nEsta ação não pode ser desfeita.",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        
        if confirm == QMessageBox.StandardButton.Yes:
            if deletar_sessao(caminho_arquivo):
                if self.conversas and self.id_sessao in caminho_arquivo:
                    self.id_sessao = f"sessao_{int(datetime.datetime.now().timestamp())}"
                    self.conversas = []
                    self.recarregar_conversas()
                    self.chat_area.append('<i>🗑️ Sessão atual excluída. Nova conversa iniciada.</i><br>')
                self.atualizar_sidebar()
        

    def carregar_sessao_sidebar(self, item):
        # Ignora o clique se o widget for um item de sessão (tratado pelo botão de deletar)
        if not item.text(): 
            arquivo_para_carregar = item.data(Qt.ItemDataRole.UserRole)
            
            # Animação de fade-out
            self.fade_out_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
            self.fade_out_animation.setDuration(200)
            self.fade_out_animation.setStartValue(1.0)
            self.fade_out_animation.setEndValue(0.0)
            self.fade_out_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
            self.fade_out_animation.finished.connect(lambda: self.finalizar_carregamento_com_fade_in(arquivo_para_carregar))
            self.fade_out_animation.start()
            return

        if "Nova Conversa" in item.text():
            self.id_sessao = f"sessao_{int(datetime.datetime.now().timestamp())}"
            self.conversas = []
            self.recarregar_conversas()
            self.chat_area.append('<i>🆕 Nova conversa iniciada.</i><br>')
        
    def finalizar_carregamento_com_fade_in(self, arquivo):
        if arquivo:
            dados = carregar_sessao(arquivo)
            self.id_sessao = dados['id']
            self.conversas = dados['conversas']
            self.recarregar_conversas()
            self.chat_area.append(f"<i>🕒 Histórico carregado da sessão: {self.id_sessao}</i><br>")
            
            # Animação de fade-in
            self.fade_in_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
            self.fade_in_animation.setDuration(300)
            self.fade_in_animation.setStartValue(0.0)
            self.fade_in_animation.setEndValue(1.0)
            self.fade_in_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
            self.fade_in_animation.start()

    def carregar_historico(self):
        sessoes = listar_sessoes()
        if not sessoes:
            return
        dlg = EscolherSessaoDialog(sessoes, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            arquivo = dlg.get_selecao()
            if arquivo:
                dados = carregar_sessao(arquivo)
                self.id_sessao = dados['id']
                self.conversas = dados['conversas']
                self.recarregar_conversas()
                self.chat_area.append(f"<i>🕒 Histórico carregado da sessão: {self.id_sessao}</i><br>")
        self.atualizar_sidebar()

    def recarregar_conversas(self):
        self.chat_area.clear()
        for c in self.conversas:
            user_html = f'<p><b><img src="resources/MelhorPresidente.png" width="36" height="36" style="border-radius: 18px; vertical-align: middle;"> Você:</b> {c["entrada"]}</p>'
            bot_html = f'<p><b><img src="resources/FazoL.png" width="36" height="36" style="border-radius: 18px; vertical-align: middle;"> Companheiro:</b> {c["resposta"]}</p>'
            self.chat_area.append(user_html)
            self.chat_area.append(bot_html)

    def send_message(self):
        if self.animating:
            return

        user_input = self.input_field.text().strip()
        if not user_input:
            return

        if user_input.lower() == "sair":
            salvar_sessao(self.id_sessao, self.conversas)
            self.close()
            return

        self.input_field.clear()

        user_html = f'<p><b><img src="resources/MelhorPresidente.png" width="36" height="36" style="border-radius: 18px; vertical-align: middle;"> Você:</b> {user_input}</p>'
        self.chat_area.append(user_html)

        # Animação de "digitando..."
        self.chat_area.append(f'<b><img src="resources/FazoL.png" width="36" height="36" style="border-radius: 18px; vertical-align: middle;"> Companheiro:</b> ...')
        QApplication.processEvents()
        
        # Simula um "pensamento" e obtém a resposta
        QTimer.singleShot(300, lambda: self.obter_e_mostrar_resposta(user_input))

    def obter_e_mostrar_resposta(self, user_input):
        # Remove o indicador "..."
        cursor = self.chat_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()
        self.chat_area.setTextCursor(cursor)
        
        # Obtém e salva a resposta
        resposta = self.chatbot.get_response(user_input)
        self.conversas.append({'entrada': user_input, 'resposta': resposta})

        # Exibe a resposta com animação
        bot_prefix = f'<b><img src="resources/FazoL.png" width="36" height="36" style="border-radius: 18px; vertical-align: middle;"> Companheiro:</b> '
        self.chat_area.append(bot_prefix)
        self.anime_texto(resposta)

    def anime_texto(self, texto):
        self.animating = True
        self.index = 0
        self.texto_anime = texto
        self.timer = QTimer()
        self.timer.timeout.connect(self.mostrar_proximo_char)
        self.timer.start(25)

    def mostrar_proximo_char(self):
        if self.index < len(self.texto_anime):
            cursor = self.chat_area.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(self.texto_anime[self.index])
            self.chat_area.setTextCursor(cursor)
            self.index += 1
        else:
            self.chat_area.append("<br>") # Adiciona quebra de linha no final
            self.timer.stop()
            self.animating = False
            self.atualizar_sidebar()

# ==========================================================
# Execução principal usando modelo .pkl
# ==========================================================

if __name__ == "__main__":
    caminho_pkl = r"C:\Users\zCarlin\Desktop\ChatBot-Teste-main\ChatBot-Teste-main\modelo_semantico.pkl"

    # Teste de carregamento do .pkl isolado
    try:
        print("🔎 Testando carregamento do modelo .pkl...")
        dados_teste = joblib.load(caminho_pkl)
        print("✅ Arquivo .pkl carregado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao carregar o arquivo .pkl: {e}")
        sys.exit(1)

    if not os.path.exists(caminho_pkl):
        print(f"❌ Arquivo .pkl não encontrado: {caminho_pkl}")
        sys.exit(1)

    chatbot_backend = Chatbot(caminho_pkl)

    app = QApplication(sys.argv)
    window = ChatbotWindow(chatbot_backend)
    window.show()
    sys.exit(app.exec())
