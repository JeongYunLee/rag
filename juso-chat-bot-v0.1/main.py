import os
import time
import uuid
from dotenv import load_dotenv
## langsmith
from langsmith import Client
from langchain_teddynote import logging
## OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import ChatMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
## ChromaDB
import chromadb
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
## LangGraph
from typing import TypedDict
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
## Streamlit
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from streamlit_feedback import streamlit_feedback
from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.tracers.langchain import wait_for_all_tracers


# .env 파일 활성화 & API KEY 설정
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')
langchain_endpoint = "https://api.smith.langchain.com"

session_id = ''

if openai_api_key:
    st.session_state["openai_api_key"] = openai_api_key
if langchain_api_key:
    st.session_state["langchain_api_key"] = langchain_api_key

# logging.langsmith("240717") 

#############################################################
########################## Title ############################
#############################################################
st.set_page_config(page_title="Address ChatBot", page_icon="🤖")
st.title("Address ChatBot")

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
            # 각 컬렉션에서 유사도 검색 수행
            documents_with_scores = collection.similarity_search_with_score(query, **self.search_kwargs)
            results.extend(documents_with_scores)
        
        # 유사도 점수를 기준으로 결과 정렬 (score가 높을수록 유사도가 높음)
        results.sort(key=lambda x: x[1], reverse=False)

        documents = [(doc, score) for doc, score in results]
        return documents

chroma_client = chromadb.PersistentClient('chroma/')
collection_names = ["csv_files_openai_3072", "49_files_openai_3072"]
embedding = OpenAIEmbeddings(model='text-embedding-3-large') 
multi_retriever = MultiCollectionRetriever(chroma_client, collection_names, embedding)
##############################################################################################################
################################################GraphState####################################################
##############################################################################################################
# GraphState 상태를 저장하는 용도
class GraphState(TypedDict):
    question: str  # 질문
    context: str  # 문서의 검색 결과
    answer: str  # llm이 생성한 답변
    relevance: str  # 답변의 문서에 대한 관련성 (groundness check)
##############################################################################################################
################################################vector Retriever##############################################
##############################################################################################################
def retrieve_document(state: GraphState) -> GraphState:
    # Question 에 대한 문서 검색을 retriever 로 수행합니다.
    retrieved_docs = multi_retriever.retrieve(state["question"])
    # 검색된 문서를 context 키에 저장합니다.
    return GraphState(context=retrieved_docs[:2])
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
            'notSure': Compare the Query and the Document. If you cannot determine whether the Document includes content that can be used to generate an answer to the Query, output the label .notSure'.
            
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
    response = chat.invoke(messages)
    return GraphState(
        relevance=response.content,
        context=state["context"],
        answer=state["answer"],
        question=state["question"],
    )

def is_relevant(state: GraphState) -> GraphState:
    if state["relevance"] == "grounded":
        return "관련성 O"
    elif state["relevance"] == "notGrounded":
        return "관련성 X"
    elif state["relevance"] == "notSure":
        return "확인불가"
##############################################################################################################
################################################LLM Answer Maker##############################################
##############################################################################################################

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

stream_handler = StreamHandler(st.empty())
chat = ChatOpenAI(model="gpt-4o", api_key=openai_api_key, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

def message(context, question):
    messages = [
        SystemMessage(content="""
            너는 Document의 정보를 반드시 활용해서 답변을 생성하는 챗봇이야. 
            이때, 답변은 Document에 정보가 있을 수도 있고, 없을 수도 있어. 
            Document의 정보로 답변을 생성할 수 있는 경우 해당 정보를 활용하고, 만약 Document의 정보로 답변을 유추조차 할 수 없는 경우, Document를 참고하지 말고 그냥 너가 생각한 답변을 생성해줘.
            주소와 관련된 질문인 경우 최대한 Document의 답변을 기반을 참고해주고, 그렇지 않은 경우 그냥 너의 지식을 활용해줘.
            답변에는 Document라는 단어를 사용하지 말아줘.
            
            답변의 끝에는 출처의 정보를 기입하는데, 출처는 Document의 'context'에 metadata의 'source'에 파일경로로 기입되어 있어. pdf, csv, md 등의 파일 이름으로만 출처를 기입해주면 돼.
            만약 여러개의 출처가 기입되어 있는 경우 모두 알려주고, 중복되는 경우 하나만 기입해줘.
            이때 파일명의 확장자(pdf, csv, md 등)는 기입하지 않아도 돼.
                      
            만약 Document를 기반으로 답변을 하지 않는 경우, 너가 아는대로 답변을 해줘. 
        """),
        HumanMessage(content=f"""
            [Document]
            {context}

            [Query]
            {question}
        """),
    ]
    return messages

def llm_answer(state: GraphState) -> GraphState:
    messages = message(state["context"], state["question"])
    response = chat.invoke(messages)
    return GraphState(
        answer=response.content,
        context=state["context"],
        question=state["question"],
    )
##############################################################################################################
################################################Setting Graph Relations#######################################
##############################################################################################################

workflow = StateGraph(GraphState)

# 노드들을 정의합니다.
workflow.add_node("retrieve", retrieve_document)  # 답변을 검색해오는 노드를 추가합니다.
workflow.add_node("llm_answer", llm_answer)  # 답변을 생성하는 노드를 추가합니다.
workflow.add_node("relevance_check", relevance_check)  # 답변의 문서에 대한 관련성 체크 노드를 추가합니다.

workflow.add_edge("retrieve", "relevance_check")  # 검색 -> 답변

# 조건부 엣지를 추가합니다.
workflow.add_conditional_edges(
    "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
    is_relevant,
    {
        "관련성 O": "llm_answer",  # 관련성이 있으면 종료합니다.
        "관련성 X": "llm_answer",  # 관련성이 없으면 다시 답변을 생성합니다.
        "확인불가": "llm_answer",  # 관련성 체크 결과가 모호하다면 다시 답변을 생성합니다.
    },
)

workflow.add_edge("llm_answer", END)  # 답변 -> 종료

# 시작점을 설정합니다.
workflow.set_entry_point("retrieve")

# 기록을 위한 메모리 저장소를 설정합니다.
memory = MemorySaver()

# 그래프를 컴파일합니다.
app = workflow.compile(checkpointer=memory)
##############################################################################################################
##############################################################################################################
##############################################################################################################
if "query" not in st.session_state:
    st.session_state.query = None

reset_history = st.sidebar.button("대화내용 초기화", type="primary")

# 메모리
msgs = StreamlitChatMessageHistory(key="langchain_messages")


if reset_history:
  msgs.clear()
  st.session_state["last_run"] = None
  st.session_state.messages = []
  st.session_state.query = None

if "messages" not in st.session_state:
  st.session_state["messages"] = []

# 세션 상태에 저장된 모든 메시지 출력
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

# 유저의 입력을 받아서 대화를 진행합니다.
if user_input := st.chat_input():
    if st.session_state.query is None:
        st.session_state.query = user_input
    st.session_state.messages.append(ChatMessage(role="user", content=user_input))
    st.chat_message("user").write(user_input)

    # RunnableConfig와 GraphState를 사용하여 답변 생성
    config = RunnableConfig(recursion_limit=5, configurable={"thread_id": "SELF-RAG"})
    inputs = GraphState(question=user_input)

    # 답변 생성 및 출력
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            output = app.invoke(inputs, config=config)
            assistant_response = output["answer"]
            st.markdown(assistant_response)
        st.session_state.messages.append(ChatMessage(role="assistant", content=assistant_response))

    # wait_for_all_tracers()
#     st.session_state.last_run = run_collector.traced_runs[0].id


# @st.cache_data(ttl="2h", show_spinner=False)
# def get_run_url(run_id):
#     time.sleep(1)
#     return client.read_run(run_id).url


# if st.session_state.get("last_run"):
#     run_url = get_run_url(st.session_state.last_run)
#     st.sidebar.markdown(f"[LangSmith 추적:hammer_and_wrench:]({run_url})")
#     feedback = streamlit_feedback(
#         feedback_type="thumbs",
#         optional_text_label="[Optional] Please provide an explanation",
#         key=f"feedback_{st.session_state.last_run}",
#     )
#     if feedback:
#         scores = {"👍": 1, "👎": 0}

#         client.create_feedback(
#             st.session_state.last_run,
#             feedback["type"],
#             score=scores[feedback["score"]],
#             value = st.session_state.query,
#             comment=feedback.get("text")
#         )
#         st.toast("피드백을 저장하였습니다.!", icon="📝")