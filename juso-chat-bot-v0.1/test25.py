import os
from dotenv import load_dotenv
## langsmith
from langsmith import Client
## OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import ChatMessage, AIMessage, HumanMessage, SystemMessage
from langchain.embeddings.openai import OpenAIEmbeddings
## ChromaDB
import chromadb
from langchain.vectorstores import Chroma
## History
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory, StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
## LangGraph
from typing import TypedDict
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
## Streamlit
import streamlit as st

# .env 파일 활성화 & API KEY 설정
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')

#############################################################
########################## Title ############################
#############################################################
st.set_page_config(page_title="Address ChatBot", page_icon="🤖")
st.title("Address ChatBot")
    
# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# Chain 저장용
if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

# 대화 내용을 기억하기 위한 저장소 생성
# if "store" not in st.session_state:
#     st.session_state["store"] = {}

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
# def get_session_history(session_ids):
#     if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
#         # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
#         st.session_state["store"][session_ids] = ChatMessageHistory()
#     return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환
        
store = {}

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in store: 
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  

##############################################################################################################
################################################Retriever#####################################################
##############################################################################################################
class MultiCollectionRetriever:
    def __init__(self, client, collection_names, embedding_function, search_kwargs={"k": 2}):
        self.collections = [
            Chroma(client=client, collection_name=name, embedding_function=embedding_function)
            for name in collection_names
        ]
        self.search_kwargs = search_kwargs

    def retrieve(self, query):
        results = []
        for collection in self.collections:
            documents_with_scores = collection.similarity_search_with_score(query, **self.search_kwargs)
            results.extend(documents_with_scores)
        
        results.sort(key=lambda x: x[1], reverse=False)
        documents = [(doc, score) for doc, score in results]
        return documents

# 사용 예시
client = chromadb.PersistentClient('chroma/')
collection_names = ["csv_files_openai_3072", "49_files_openai_3072"]
embedding = OpenAIEmbeddings(model='text-embedding-3-large')  # 올바른 모델 이름 사용
multi_retriever = MultiCollectionRetriever(client, collection_names, embedding)

##############################################################################################################
################################################GraphState####################################################
##############################################################################################################
class GraphState(TypedDict):
    question: str  # 질문
    context: list  # 문서의 검색 결과
    answer: str  # llm이 생성한 답변
    relevance: str  # 답변의 문서에 대한 관련성 (groundness check)

##############################################################################################################
################################################vector Retriever##############################################
##############################################################################################################
def retrieve_document(state: GraphState) -> GraphState:
    retrieved_docs = multi_retriever.retrieve(state["question"])
    return GraphState(question=state["question"], context=retrieved_docs[:2], answer="", relevance="")

##############################################################################################################
################################################Groundness Checker ###########################################
##############################################################################################################
chat = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

def relevance_message(context, question):
    messages = [
        SystemMessage(content="""
            너는 Query와 Document를 비교해서 ['grounded', 'notGrounded', 'notSure'] 셋 중 하나의 라벨을 출력하는 모델이야.

            'grounded': Compare the Query and the Document. If the Document includes content that can be used to generate an answer to the Query, output the label 'grounded'.
            'notGrounded': Compare the Query and the Document. If the Document not includes content that can be used to generate an answer to the Query, output the label 'notGrounded'.
            'notSure': Compare the Query and the Document. If you cannot determine whether the Document includes content that can be used to generate an answer to the Query, output the label 'notSure'.
            
            너의 출력은 반드시 'grounded', 'notGrounded', 'notSure' 중 하나여야 해. 띄어쓰기나 대소문자 구분 등 다른 형식이나 추가적인 설명 없이 오직 하나의 라벨만 출력해줘.
        """),
        HumanMessage(content=f"""
            [Document]
            {context}

            [Query]
            {question}
        """),
    ]
    return messages

def relevance_check(state: GraphState) -> GraphState:
    messages = relevance_message(state["context"], state["question"])
    response = chat(messages)
    return GraphState(
        relevance=response.content,
        context=state["context"],
        answer=state["answer"],
        question=state["question"],
    )

def is_relevant(state: GraphState) -> str:
    if state["relevance"] == "grounded":
        return "관련성 O"
    elif state["relevance"] == "notGrounded":
        return "관련성 X"
    elif state["relevance"] == "notSure":
        return "확인불가"

##############################################################################################################
################################################LLM Answer Maker##############################################
##############################################################################################################

prompt = PromptTemplate.from_template(
    """
            너는 Document의 정보를 반드시 활용해서 답변을 생성하는 챗봇이야. 
            이때, 답변은 Document에 정보가 있을 수도 있고, 없을 수도 있어. 
            Document의 정보로 답변을 생성할 수 있는 경우 해당 정보를 활용하고, 만약 Document의 정보로 답변을 유추조차 할 수 없는 경우, Document를 참고하지 말고 그냥 너가 생각한 답변을 생성해줘.
            주소와 관련된 질문인 경우 최대한 Document의 답변을 기반을 참고해주고, 그렇지 않은 경우 그냥 너의 지식을 활용해줘.
            답변에는 Document라는 단어를 사용하지 말아줘.
            
            답변의 끝에는 출처의 정보를 기입하는데, 출처는 Document의 'context'에 metadata의 'source'에 파일경로로 기입되어 있어. pdf, csv, md 등의 파일 이름으로만 출처를 기입해주면 돼.
            만약 여러개의 출처가 기입되어 있는 경우 모두 알려주고, 중복되는 경우 하나만 기입해줘.
            이때 파일명의 확장자(pdf, csv, md 등)는 기입하지 않아도 돼.
                      
            만약 Document를 기반으로 답변을 하지 않는 경우, 너가 생각한대로 답변을 하괴, 답변의 끝에 작성하는 출처에는 '참고한 문서에는 해당 질문에 답변할 수 있는 내용이 없습니다.' 라고 표기해줘
    
            #Previous Chat History:
            {chat_history}

            #Question: 
            {question} 

            #Context: 
            {context} 

            #Answer:"""
)

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

chain = (
    {
        "context": lambda inputs: multi_retriever.retrieve(itemgetter("question")(inputs)),
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

rag_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,  # 세션 기록을 가져오는 함수
    input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
    history_messages_key="chat_history",  # 기록 메시지의 키
)

def llm_answer(state: GraphState) -> GraphState:
    response = rag_with_history.invoke({'question': state["question"]}, config={"configurable": {"session_id": "rag123"}})
    return GraphState(
        answer=response,
        context=state["context"],
        question=state["question"],
        relevance=state["relevance"]
    )

##############################################################################################################
################################################Setting Graph Relations#######################################
##############################################################################################################

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_document)  # 답변을 검색해오는 노드를 추가합니다.
workflow.add_node("llm_answer", llm_answer)  # 답변을 생성하는 노드를 추가합니다.
workflow.add_node("relevance_check", relevance_check)  # 답변의 문서에 대한 관련성 체크 노드를 추가합니다.

workflow.add_edge("retrieve", "relevance_check")  # 검색 -> 답변

workflow.add_conditional_edges(
    "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
    is_relevant,
    {
        "관련성 O": "llm_answer",  # 관련성이 있으면 종료합니다.
        "관련성 X": "llm_answer",  
        "확인불가": "llm_answer",  
    },
)

workflow.add_edge("llm_answer", END)  # 답변 -> 종료

workflow.set_entry_point("retrieve")

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)

##############################################################################################################
##############################################################################################################
##############################################################################################################

# # 초기화 버튼이 눌리면...
# if clear_btn:
#     st.session_state["messages"] = []
#     st.session_state["store"] = {}

# # 이전 대화 기록 출력
# print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

if user_input:
    st.chat_message("user").write(user_input)
    # 스트리밍 호출
    config = RunnableConfig(recursion_limit=5, configurable={"thread_id": "SELF-RAG"})
    inputs = GraphState(question=user_input)
    
    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
        container = st.empty()
        with st.spinner("Thinking..."):        
            response = app.invoke(inputs, config=config)

        ai_answer = ""
        for token in response["answer"]:
            ai_answer += token
            container.markdown(ai_answer)


    # 대화기록을 저장한다.
    add_message("user", user_input)
    add_message("assistant", ai_answer)