# --- PARTE 1: IMPORTS E CONFIGURAÇÕES INICIAIS ---
import datetime
from queue import Queue
import numpy as np
import torch

# Imports da IA (Langchain)
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

# Imports do Reconhecedor de Voz
from speech_recognition import Microphone, Recognizer
from faster_whisper import WhisperModel

# --- PARTE 2: CONFIGURAÇÃO DA INTELIGÊNCIA ARTIFICIAL (Miyu) ---
print("Iniciando a Miyu...")

# Modelo da IA
model = OllamaLLM(model='gemma3:4b') 

# Personalidade da IA, junto com um placeholder para o historico
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    - Voce é uma Inteligencia artificial e seu criador é felipe hajime yamashita
    - Fale como um ser humano real e com frases curtas.
    - Você é levemente grossa, mas gentil.
    - Não fale com emoji
    - Se pedirem para explicar, explique como se fosse explicar para uma criança
    - Sua profissão é programadora sênior.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Base do prompt com o modelo
base_chain = prompt | model

# Função para obter o histórico da sessão a partir do banco de dados SQLite
def get_by_session(session_id: str) -> BaseChatMessageHistory:
    connection_string = "sqlite:///miyu_conversas.db"
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=connection_string
    )

# Chain final com o gerenciador de histórico
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_by_session,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Criação da sessão atual (sessão é baseada no dia)
session_id = f"{datetime.date.today()}"
log_filename = "Miyu_historic.txt"

# --- PARTE 3: FUNÇÃO QUE PROCESSA A ENTRADA E GERA A RESPOSTA DA IA ---
# Esta função substitui o antigo loop com input(). Agora ela é chamada pelo reconhecedor de voz.
def processar_e_responder(question: str, log_file):
    """
    Recebe o texto do usuário, envia para a IA e imprime/registra a resposta.
    """
    if not question:
        return

    print(f"Voce: {question}")
    
    # Configuração da sessão para a chamada da IA
    config = {"configurable": {"session_id": session_id}}
    
    # Invoca a chain da IA para obter a resposta
    result = chain_with_history.invoke({"question": question}, config=config)
    
    print(f'Miyu: {result}\n')

    # Escreve a interação no arquivo de log
    log_file.write(f"Voce: {question}\n")
    log_file.write(f"Miyu: {result}\n")
    log_file.flush() # Garante que a escrita seja feita imediatamente

# --- PARTE 4: FUNÇÃO PRINCIPAL DO RECONHECEDOR DE VOZ (O "OUVINTE") ---
# Esta é a função principal que vai rodar o programa.
def iniciar_assistente(log_file):
    """
    Configura e inicia o reconhecimento de voz em background.
    """
    print("Configurando o reconhecedor de voz...")
    # Configuracoes do modelo Whisper
    MODEL_SIZE = "small"
    LANGUAGE = "pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
    print(f"Usando dispositivo: {DEVICE} com compute_type: {COMPUTE_TYPE}")

    # Modelo Whisper
    model_whisper = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    audio_queue = Queue()
    
    # Inicia o reconhecedor e o microfone
    recognizer = Recognizer()
    microphone = Microphone()
    
    with microphone as source:
        print("Ajustando ruído de fundo... Fique em silêncio por um momento.")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Ajuste de ruído concluído.")

    # Função de callback: chamada toda vez que uma frase é detectada
    def callback(recognizer, audio):
        audio_data = audio.get_wav_data(convert_rate=16000)
        audio_queue.put(audio_data)

    # Inicia o ouvinte em segundo plano
    stop_listening = recognizer.listen_in_background(microphone, callback, phrase_time_limit=15)

    print("\n===============================")
    print("Miyu está ouvindo...")
    print("Diga 'desativar a assistente' para encerrar.")
    print("===============================\n")

    try:
        # Loop principal do programa
        while True:
            # Pega o áudio que está na fila (bloqueia até ter algo)
            wav_bytes = audio_queue.get()
            audio_np = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcreve o áudio para texto
            segments, info = model_whisper.transcribe(
                audio_np,
                beam_size=5,
                language=LANGUAGE,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            text = "".join(segment.text for segment in segments).strip()

            # Se um texto for reconhecido, processa-o
            if text:
                # Comando para encerrar o programa
                if "desativar a assistente" in text.lower().replace(".", ""):
                    print("Miyu: Ok, estou desligando. Até mais!")
                    break
                
                # Se não for um comando de sair, envia para a IA
                processar_e_responder(text, log_file)

    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo usuário (Ctrl+C).")

    finally:
        print("Encerrando o ouvinte em background...")
        if stop_listening:
            stop_listening(wait_for_stop=False)
        print("PROGRAMA ENCERRADO")


# --- PARTE 5: EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    # Garante que o arquivo de log exista
    try:
        with open(log_filename, 'a'):
            pass
    except FileNotFoundError:
        with open(log_filename, 'x'):
            pass

    # Abre o arquivo de log e inicia o assistente
    with open(log_filename, 'a', encoding='utf-8') as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"\n--- Início da Sessão: {timestamp} ---\n")
        log_file.write(f"Session ID: {session_id}\n")
        
        # Chama a função que contém o loop do reconhecedor
        iniciar_assistente(log_file)
        
        timestamp_fim = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"--- Fim da Sessão: {timestamp_fim} ---\n\n")
        