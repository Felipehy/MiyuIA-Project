from speech_recognition import Microphone,Recognizer
from faster_whisper import WhisperModel
from queue import Queue
from MiyuIA import miyu_assistent
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
import datetime
import os
import numpy as np
import torch



#Funcao para ouvir e imprimir
def main():

    #Controllers
    isMiyu = False

    #Configuracoes do modelo
    MODEL_SIZE = "small"
    LINGUAGE = "pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

    #Modelo 
    model = WhisperModel(MODEL_SIZE,device=DEVICE,compute_type=COMPUTE_TYPE)
    audio_queue = Queue()
    
    #Inicia o reconhecedor e o microfone e ja faz o ajuste do ruido
    recognizer = Recognizer()
    microphone = Microphone()
    
    with microphone as source:
        print("Esta configurando o audio fique em silencio")
        recognizer.adjust_for_ambient_noise(source, duration=3)
        
    #Toda vez que escutar uma frase ela será chamada e vai guardar a frase para ser processada na fila
    def callback(recognizer, audio):
        audio_data = audio.get_wav_data(convert_rate=16000)
        audio_queue.put(audio_data)

    stop_listening = recognizer.listen_in_background(microphone, callback, phrase_time_limit=15)

    print("ASSISTENTE INICIADO")
    try:
        #variaveis para o log
        session_id = f"{datetime.date.today()}"
        log_filename = "Miyu_historic.txt"
        # Faz a aberto do documento log
        with open(log_filename, 'a', encoding='utf-8') as log_file:
            timeStamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"\n--- Início da Conversa: {timeStamp} ---\n")
            log_file.write(f"Session ID: {session_id}\n")

        while True:
            #pega o audio que esta na fila
            wav_bytes = audio_queue.get()

            audio_np = np.frombuffer(wav_bytes,dtype=np.int16).astype(np.float32) / 32768.0

            segments, info = model.transcribe(
                audio_np,
                beam_size=5,
                language=LINGUAGE,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            text = "".join(segment.text for segment in segments).strip()

            if text:
                print(f"Voce: {text}")

                if "desativar a assistente" in text.lower().replace(".",""):
                    break
                elif "ferramentas" in text.lower().replace(".",""):
                    #Abrir ferramentas da assistente
                    #tools(text)
                    pass
                elif "iniciar ia" in text.lower().replace(".",""):
                    #vai iniciar a MiyuIA
                    isMiyu = True
                    pass

                if isMiyu:
                    miyuConverse = miyu_assistent(text, session_id, log_file, timeStamp)
                    if miyuConverse == "desativar ia":
                        isMiyu = False
                    print(miyuConverse)
            else:
                print("Nao entendi fala de novo")

            

    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo usuário (Ctrl+C).")

    finally:
        print("Parando o programa background")
        if stop_listening:
            stop_listening(wait_for_stop=False)
        print("PROGRAMA ENCERRADO")


if __name__ == "__main__":
    main()
