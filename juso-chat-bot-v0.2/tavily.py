import os
import pprint
from dotenv import load_dotenv
## langsmith
from langsmith import Client
from langchain_teddynote import logging
## OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import ChatMessage, AIMessage, HumanMessage, SystemMessage
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
## ChromaDB
import chromadb
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
## History
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory, StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
## LangGraph
from typing import TypedDict
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langchain_community.tools.tavily_search import TavilySearchResults
## Google Search
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
## gradio
import gradio as gr

# .env 파일 활성화 & API KEY 설정
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

logging.langsmith("240731") 

store = {}

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환
    
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
        return "grounded"
    elif state["relevance"] == "notGrounded":
        return "notGrounded"
    elif state["relevance"] == "notSure":
        return "notSure"
    
##############################################################################################################
################################################# Rewriter ###################################################
##############################################################################################################  
    
def rewrite(state):
    question = state["question"]
    answer = state["answer"]
    context = state["context"]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional prompt rewriter. Your task is to generate the question in order to get additional information that is now shown in the context."
                "Your generated question will be searched on the web to find relevant information.",
            ),
            (
                "human",
                "Rewrite the question to get additional information to get the answer."
                "\n\nHere is the initial question:\n ------- \n{question}\n ------- \n"
                "\n\nHere is the initial context:\n ------- \n{context}\n ------- \n"
                "\n\nHere is the initial answer to the question:\n ------- \n{answer}\n ------- \n"
                "\n\nFormulate an improved question in Korean:",
            ),
        ]
    )

    # Question rewriting model
    model = ChatOpenAI(temperature=0, model="gpt-4o")

    chain = prompt | model | StrOutputParser()
    response = chain.invoke(
        {"question": question, "answer": answer, "context": context}
    )
    return GraphState(question=response)

    
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
################################################LLM Answer Maker##############################################
##############################################################################################################

def llm_answer(state: GraphState) -> GraphState:
    
    # 프롬프트를 생성합니다.
    prompt = PromptTemplate.from_template(
            """
                당신은 Context의 정보를 활용하여 답변을 생성하는 챗봇입니다. 답변을 생성할 때는 다음의 지침을 따르세요:

                    1.	정보 활용: Context에 관련된 정보가 있는 경우, 해당 정보를 기반으로 답변을 생성하세요.
                    2.	정보 부족 시: Context의 정보로 답변을 유추할 수 없는 경우, 일반적인 지식에 따라 답변을 생성하세요. 이때 Context는 참고하지 말고, 단순히 당신이 알고 있는 내용을 바탕으로 답변을 작성하세요.
                    3.	주소 관련 질문: 주소와 관련된 질문이 있을 경우, 최대한 Context의 정보를 바탕으로 답변을 제공하세요. 다른 주제의 질문에는 Context를 활용하지 않아도 됩니다.

                출처 기입:

                    - 답변의 끝에는 출처를 기입해야 합니다.
                    - 출처는 Context의 ‘metadata’에 있는 ‘source’에서 파일 이름을 기준으로 작성합니다. 파일 확장자(pdf, csv, md 등)와 파일 경로(data/final, data/csv 등)는 기입하지 마세요.
                    - 만약 출처가 여러 개인 경우, 모두 기입하되 중복된 출처는 하나만 적어주세요.
                    - Context에 ‘metadata’의 ‘source’가 없으면 출처를 적지 않습니다.
                    - 답변에 'Context'라는 단어는 표시하지 않습니다. 
                    - Context를 기준으로 답변을 생성하지 않았다면, 출처는 기입하지 않습니다. 
                
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

    print(state['context'])
    
    response = rag_with_history.invoke(input_data, config={"configurable": {"session_id": "rag123"}})
    
    return GraphState(
        answer=response,
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
workflow.add_node("rewrite", rewrite)  # 질문을 재작성하는 노드를 추가합니다.
workflow.add_node("search_on_web", search_on_web)  # 웹 검색 노드를 추가합니다.

workflow.add_edge("retrieve", "relevance_check")  # 검색 -> 답변
workflow.add_edge("rewrite", "search_on_web")  # 재작성 -> 관련성 체크
workflow.add_edge("search_on_web", "relevance_check")  # 웹 검색 -> 답변

# 조건부 엣지를 추가합니다.
workflow.add_conditional_edges(
    "relevance_check",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
    is_relevant,
    {
        "grounded": "llm_answer",  # 관련성이 있으면 종료합니다.
        "notGrounded": "rewrite",  # 관련성이 없으면 다시 질문을 작성합니다.
        "notSure": "rewrite",  # 관련성 체크 결과가 모호하다면 다시 질문을 작성합니다.
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
################################################Chat Interface################################################
##############################################################################################################

def source(result):
    try:
        source_list = [item['url'] for item in result['context']]
        source = '자세한 정보: ' + str(source_list).replace('[', '').replace(']', '').replace("'", '') 
    except:
        source = "not link"
    return source

def stream_responses(question, history):
    config = RunnableConfig(
        recursion_limit=12, configurable={"thread_id": "CORRECTIVE-SEARCH-RAG"}
    )

    # AgentState 객체를 활용하여 질문을 입력합니다.
    inputs = GraphState(
        question=question
    )

    # app.stream을 통해 입력된 메시지에 대한 출력을 스트리밍합니다.
    try:
        output_generator = app.invoke(inputs, config=config)
        source_item = source(output_generator)
        
        if source_item == "not link":
            answer = output_generator['answer']
        else:
            answer = output_generator['answer'] + source_item
    except GraphRecursionError as e:
        # 예외 발생 시 반환값을 명확히 설정합니다.
        print(f"Recursion limit reached: {e}")
        answer = "죄송합니다. 해당 질문에 대해서는 답변할 수 없습니다."
    return answer
    
gr.ChatInterface(
    stream_responses,
    chatbot=gr.Chatbot(),
    textbox=gr.Textbox(placeholder="", container=False, scale=7),
    title="Address Chatbot",
    description="Ask questions about Korean Address.",
    theme="soft",
    examples=["주소와 주소정보의 차이점은?", "건물번호에 대해서 알려줘"],
    cache_examples=False,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch()