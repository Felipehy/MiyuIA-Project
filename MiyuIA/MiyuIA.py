from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
import datetime


#Modelo da IA
model = OllamaLLM(model='gemma3:4b') 

#Personalidade da IA, junto com um placeholder para o historico
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

#Base do prompt com o modelo
base_chain = prompt | model

#memoria para o RunnableWithMessageHistory
store = {}

#Id da sessao com integracao com SQLlite
def get_by_session(session_id: str) -> BaseChatMessageHistory:
    connection_string = "sqlite:///miyu_conversas.db"

    return SQLChatMessageHistory(
        session_id=session_id,
        connection=connection_string
    )

# chain com o gerenciador do historico
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_by_session,
    input_messages_key="question",
    history_messages_key="chat_history",
)

#Ver se existe ja uma Miyu_historic.txt
def isFileExist():
    try:
        file = open('Miyu_historic.txt')
        file.close()
        return True
    except:
        return False

#verifica se existe um txt, se nao criar um arquivo novo
if not isFileExist():
    #print("Criando arquivo txt")
    f = open('Miyu_historic.txt',"x")
    f.close()

def miyu_assistent(text, session_id, log_file, timeStamp):

        if text.lower() == 'desativar ia':
            log_file.write(f'--- Fim da Conversa: {timeStamp} ---\n')
            log_file.write(f'Session_id: {session_id}')
            return text
        
        config = {"configurable": {"session_id": session_id}}
        result = chain_with_history.invoke({"question": text}, config=config)
        
        log_file.write(f"Voce: {text}")
        log_file.write(f"Miyu: {result}")
        log_file.flush()

        return f'Miyu: {result}'

