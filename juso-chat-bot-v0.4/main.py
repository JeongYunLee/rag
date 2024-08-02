import os
import pprint
from dotenv import load_dotenv
## langsmith
from langsmith import Client
from langchain_teddynote import logging
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
from langgraph.errors import GraphRecursionError
from langchain_community.tools.tavily_search import TavilySearchResults
## Google Search
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
## chainlit
import chainlit as cl
from typing import Dict, Optional
## literaliteral_client
from literalai import LiteralClient

# .env 파일 활성화 & API KEY 설정
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

logging.langsmith("240731") 

literal_client = LiteralClient(api_key=os.getenv("LITERAL_API_KEY"))
literal_client.instrument_openai()

##############################################################################################################
##############################################################################################################
##############################################################################################################

admin_id = os.getenv("ADMIN_ID")
admin_pw = os.getenv("ADMIN_PW")

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == (admin_id, admin_pw):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.oauth_callback
def oauth_callback(
  provider_id: str,
  token: str,
  raw_user_data: Dict[str, str],
  default_user: cl.User,
) -> Optional[cl.User]:
  return default_user


##############################################################################################################
################################################GraphState####################################################
##############################################################################################################
# GraphState 상태를 저장하는 용도
class GraphState(TypedDict):
    question: str  # 질문
    q_type: str # 질문의 유형
    context: str  # 문서의 검색 결과
    answer: str  # llm이 생성한 답변
    relevance: str  # 답변의 문서에 대한 관련성 (groundness check)
    
store = {}

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

##############################################################################################################
################################################Decision Maker ###############################################
##############################################################################################################

def decision_maker(state: GraphState) -> GraphState:
    chat = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
    
    prompt = PromptTemplate.from_template(
         """
            너는 question의 종류를 분류하는 모델이야. 질문의 종류는 ['주소관련 질문', '검색필요 질문', '일반 질문'] 3가지 종류로 구분 돼. 이때 question뿐만 아니라 chat_history까지 고려해줘.
            
            1. 주소관련 질문: 주소(address)와 관련된 질문을 의미해. 예를 들어, 주소와 관련된 개념, 정의, 주소관련 데이터 분석 등과 같은 내용의 질문일 경우 '주소관련 질문'으로 분류해 줘.
            2. 검색필요 질문: 주소(address)와 관련되지 않은 질문 중, 너가 스스로 대답할 수 없는 질문을 의미해. 예를 들어, '오늘의 날씨', '오늘의 주가' 등과 같이 최신 정보를 반영해야 하는 경우와 '~검색해줘'라는 말이 포함될 때 '검색필요 질문'으로 분류해 줘.
            3. 일반 질문: 주소(address)와 관련되지 않은 질문 중, 너가 스스로 대답할 수 있는 질문을 의미해. 예를 들어, '영어를 한국어로 번역해줘', '대한민국의 수도는?'과 같이 일반 상식적인 질문인 경우 '일반 질문'으로 분류해 줘.
            
            질문이 들어왔을 때, 위 3개의 종류 중에 가장 해당되는 분류를 선택하고 반드시 ['주소관련 질문', '검색필요 질문', '일반 질문'] 중 하나로 선택해. 띄어쓰기나 대소문자 구분 등 다른 형식이나 추가적인 설명 없이 오직 하나의 라벨만 출력해줘.

            #Previous Chat History:
            {chat_history}

            #Question: 
            {question} 
            #Answer:"""
            )
    
    chain = prompt | chat | StrOutputParser()
    
    rag_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    
    input_data = {
        'question': state["question"],
        'chat_history': itemgetter("chat_history"),
        'context':''
    }
    
    response = rag_with_history.invoke(input_data, config={"configurable": {"session_id": "rag123"}})
    return GraphState(
        q_type=response,
        question=state["question"],
    )

def decision_making(state: GraphState) -> GraphState:
    q_type_strip = state["q_type"].strip()
    if q_type_strip == "주소관련 질문":
        return "about_address"
    elif q_type_strip == "검색필요 질문":
        return "search"
    elif q_type_strip == "일반 질문":
        return "general"
    

##############################################################################################################
################################################Groundness Checker ###########################################
##############################################################################################################

chat = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)

def relevance_message(context, question):
    
    messages = [
        SystemMessage(content="""
            너는 Query와 Document를 비교해서 ['grounded', 'notGrounded', 'notSure'] 셋 중 하나의 라벨을 출력하는 모델이야.

            'grounded': Compare the Query and the Document. If the Document includes content that can be used to generate an answer to the Query, output the label 'grounded'.
            'notGrounded': Compare the Query and the Document. If the Document does not include content that can be used to generate an answer to the Query, or if the information is insufficient, output the label ‘notGrounded’.
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
    print(messages)
    print(state['context'])
    
    response = chat.invoke(messages)
    return GraphState(
        relevance=response.content,
        context=state["context"],
        answer=state["answer"],
        question=state["question"],
    )

def is_relevant(state: GraphState) -> GraphState:
    relevance_strip = state["relevance"].strip()
    if relevance_strip == "grounded":
        return "grounded"
    elif relevance_strip == "notGrounded":
        return "notGrounded"
    elif relevance_strip == "notSure":
        return "notSure"
    

##############################################################################################################
################################################LLM Answer Maker##############################################
##############################################################################################################
def llm_answer(state: GraphState) -> GraphState:
    
    # 프롬프트를 생성합니다.
    prompt = PromptTemplate.from_template(
        """
                너는 Context의 정보를 반드시 활용해서 답변을 생성하는 챗봇이야. 
                이때, 답변은 Context에 정보가 있을 수도 있고, 없을 수도 있어. 
                Context의 정보로 답변을 생성할 수 있는 경우 해당 정보를 활용하고, 만약 Context의 정보로 답변을 유추조차 할 수 없는 경우, Context를 참고하지 말고 잘 모르겠다고 답변해줘.
                답변에는 Context라는 단어를 사용하지 말아줘.
                
                만약 Context의 정보를 활용한다면, 출처를 알 수 있는 경우 출처(data/final, data/csv 등 뒤에 있는 파일명 혹은 'http:', 'https' 등으로 시작하는 url)를 마지막에 넣어줘. 하지만 잘 모르겠다면 빼도 돼. 
                파일명을 출처로 작성한다면 'data/final', 'data/csv' 와 같은 경로명이나 .csv, .docx, .pdf, .txt 등 파일 경로와 확장자는 제거해줘. 

                #Previous Chat History:
                {chat_history}

                #Question: 
                {question} 

                #Context: 
                {context} 

                #Answer:
                
            """
                )

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    # 프롬프트, 모델, 출력 파서를 체이닝합니다.
    chain = prompt | llm | StrOutputParser()

    # 대화를 기록하는 RAG 체인 생성
    rag_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    
    # 상태에서 질문과 대화 기록을 가져옵니다.
    input_data = {
        'question': state["question"],
        'chat_history': itemgetter("chat_history"),
        'context': state["context"]
    }

    response = rag_with_history.invoke(input_data, config={"configurable": {"session_id": "rag123"}})
    
    return GraphState(
        answer=response,
        context=state["context"],
        question=state["question"],
    )


##############################################################################################################
#############################################General LLM Answer Maker#########################################
##############################################################################################################
def general_llm(state: GraphState) -> GraphState:
    
    # 프롬프트를 생성합니다.
    prompt = PromptTemplate.from_template(
        """     
                너는 일반적인 상식과 정보에 대해서 답변하는 챗봇이야. chat_hisotry에 값이 있는 경우 그 값을 참고해서 답변을 해줘.

                #Previous Chat History:
                {chat_history}

                #Question: 
                {question} 
                
                #Answer:
                
            """
                )

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    # 프롬프트, 모델, 출력 파서를 체이닝합니다.
    chain = prompt | llm | StrOutputParser()

    # 대화를 기록하는 RAG 체인 생성
    rag_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    
    # 상태에서 질문과 대화 기록을 가져옵니다.
    input_data = {
        'question': state["question"],
        'chat_history': itemgetter("chat_history"),
    }

    response = rag_with_history.invoke(input_data, config={"configurable": {"session_id": "rag123"}})
    return GraphState(
        answer=response,
        question=state["question"],
    )
    
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

# 사용 예시
client = chromadb.PersistentClient('../juso-chat-bot-v0.1/chroma/')
collection_names = ["csv_files_openai_3072", "49_files_openai_3072"]
embedding = OpenAIEmbeddings(model='text-embedding-3-large') 
multi_retriever = MultiCollectionRetriever(client, collection_names, embedding)

##############################################################################################################
################################################vector Retriever##############################################
##############################################################################################################
def retrieve_document(state: GraphState) -> GraphState:
    # Question 에 대한 문서 검색을 retriever 로 수행합니다.
    retrieved_docs = multi_retriever.retrieve(state["question"])
    # 검색된 문서를 context 키에 저장합니다.
    return GraphState(context=retrieved_docs[:2])

##############################################################################################################
################################################Search on Web ################################################
##############################################################################################################  
    
def search_on_web(state: GraphState) -> GraphState:
    # 문서에서 검색하여 관련성 있는 문서를 찾습니다.
    search_tool = TavilySearchResults(max_results=5)
    search_result = search_tool.invoke({"query": state["question"]})

    # 검색된 문서를 context 키에 저장합니다.
    return GraphState(
        context=search_result,
    )
    
##############################################################################################################
################################################Setting Graph Relations#######################################
##############################################################################################################

workflow = StateGraph(GraphState)

# 노드들을 정의합니다.
workflow.add_node("decision_maker", decision_maker)  # 질문의 종류를 분류하는 노드를 추가합니다.
workflow.add_node("retrieve", retrieve_document)  # 답변을 검색해오는 노드를 추가합니다.
workflow.add_node("general_llm", general_llm)  # 일반 질문에 대한 답변을 생성하는 노드를 추가합니다.
workflow.add_node("relevance_check", relevance_check)  # 답변의 문서에 대한 관련성 체크 노드를 추가합니다.
workflow.add_node("search_on_web", search_on_web)  # 웹 검색 노드를 추가합니다.
workflow.add_node("llm_answer", llm_answer)  # 답변을 생성하는 노드를 추가합니다.

workflow.add_edge("retrieve", "relevance_check")  # 검색 -> 답변
workflow.add_edge("search_on_web", "relevance_check")  # 웹 검색 -> 답변

# 조건부 엣지를 추가합니다.
workflow.add_conditional_edges(
    "decision_maker",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
    decision_making,
    {
        "about_address": "retrieve",  # 관련성이 있으면 종료합니다.
        "search": "search_on_web",  # 관련성이 없으면 다시 질문을 작성합니다.
        "general": "general_llm",  # 관련성 체크 결과가 모호하다면 다시 질문을 작성합니다.
    },
)

# 조건부 엣지를 추가합니다.
workflow.add_conditional_edges(
    "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
    is_relevant,
    {
        "grounded": "llm_answer",  # 관련성이 있으면 종료합니다.
        "notGrounded": "search_on_web",  # 관련성이 없으면 다시 질문을 작성합니다.
        "notSure": "search_on_web",  # 관련성 체크 결과가 모호하다면 다시 질문을 작성합니다.
    },
)

workflow.add_edge("llm_answer", END)  # 답변 -> 종료
workflow.add_edge("general_llm", END)  # 답변 -> 종료

# 시작점을 설정합니다.
workflow.set_entry_point("decision_maker")

# 기록을 위한 메모리 저장소를 설정합니다.
memory = MemorySaver()

# 그래프를 컴파일합니다.
app = workflow.compile(checkpointer=memory)

##############################################################################################################
################################################Chat Interface################################################
##############################################################################################################

@cl.on_message
async def run_convo(message: cl.Message):
    
    config = RunnableConfig(
        recursion_limit=20, thread_id = 'CORRECTIVE-SEARCH-RAG',
    )

    # AgentState 객체를 활용하여 질문을 입력합니다.
    inputs = GraphState(
        question=message.content
    )

    # app.stream을 통해 입력된 메시지에 대한 출력을 스트리밍합니다.   
    try:
        answer = app.invoke(inputs, config=config)
    except GraphRecursionError as e:
        # 예외 발생 시 반환값을 명확히 설정합니다.
        print(f"Recursion limit reached: {e}")
        answer = "죄송합니다. 해당 질문에 대해서는 답변할 수 없습니다."


    await cl.Message(content=answer['answer']).send()