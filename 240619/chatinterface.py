from dotenv import load_dotenv
import os
import uuid
import chromadb
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from chromadb.utils import embedding_functions
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from kiwipiepy import Kiwi
import gradio as gr
import time


# .env 파일 활성화
load_dotenv()

from kiwipiepy import Kiwi

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

############################################################################################################
######### 파일 불러오기 #######################################################################################
############################################################################################################

loader = DirectoryLoader(path='data', glob='*.txt', loader_cls=TextLoader, show_progress=True, use_multithreading=True)
data = loader.load()

############################################################################################################
######### 컬렉션 생성/연결하기 ##################################################################################
############################################################################################################
client = chromadb.PersistentClient('chroma/')
collection = client.get_or_create_collection(name="ten_files_openai_3072")

############################################################################################################
############# txt doc 설정하기 ################################################################################
############################################################################################################
doc_ids = [str(uuid.uuid4()) for _ in data]
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 200)

id_key = "doc_id"

child_docs = []  # 하위 문서를 저장할 리스트를 초기화합니다.
for i, doc in enumerate(data):
    _id = doc_ids[i]  # 현재 문서의 ID를 가져옵니다.
    # 현재 문서를 하위 문서로 분할합니다.
    child_doc = child_text_splitter.split_documents([doc])
    for _doc in child_doc:  # 분할된 하위 문서에 대해 반복합니다.
        # 하위 문서의 메타데이터에 ID를 저장합니다.
        _doc.metadata[id_key] = _id
    child_docs.extend(child_doc)  # 분할된 하위 문서를 리스트에 추가합니다.

############################################################################################################
############# 검색하기 #######################################################################################
############################################################################################################

kiwi = Kiwi()

def kiwi_tokenize(text):
    return [token.form for token in kiwi.tokenize(text)]


# BM25 retrieval 
bm25_retriever = BM25Retriever.from_documents(child_docs) #preprocess_func = kiwi_tokenize
bm25_retriever.k = 1  # BM25Retriever의 검색 결과 개수를 1로 설정합니다.

#DB retrieval 
embedding = OpenAIEmbeddings(model='text-embedding-3-large')  # OpenAI 임베딩을 사용합니다.

vectorstore = Chroma(client=client, collection_name="ten_files_openai_3072", embedding_function=embedding)
chromadb_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Ensemble retrieval 
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chromadb_retriever],
    weights=[0.5, 0.5],
    search_type="mmr",
)

############################################################################################################
############# Gradio 인터페이스 생성 #############################################################################
############################################################################################################

def chat(query, history):
    ensemble_result = ensemble_retriever.invoke(query)
    max_similarity_doc = ensemble_result[0].page_content + '\n######\n' + ensemble_result[1].page_content

    chat = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)


    messages = [
        SystemMessage(content=f"""
                    너는 Document의 정보를 반드시 활용해서 답변을 생성하는 챗봇이야. 
                    Document의 정보로 답변을 할 수 없는 경우, 새로운 정보를 생성하지 말고 '정보가 부족해서 답을 할 수 없습니다' 라고 답변해줘.
                    답변에는 Document라는 단어를 사용하지 말아줘.
                    """),

        HumanMessage(content=f"""
                    [Document]
                    {max_similarity_doc}

                    [Query]
                    {query}
                    """),
    ]

    output = chat.invoke(messages)

    # answer = output.content + "\n\n ====================<Resource>====================\n\n" + max_similarity_doc

    return output.content

# Gradio 인터페이스 설정
# iface = gr.Interface(
#     fn=chat,
#     inputs="text",
#     outputs=["text", "text"],
#     title="Chat with Documents",
#     description="Ask questions and get answers based on the provided documents.",
#     allow_flagging="auto",
# )


gr.ChatInterface(
    chat,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="", container=False, scale=7),
    title="Chat with document",
    description="Ask questions and get answers based on the provided documents.",
    theme="soft",
    examples=["주소와 주소정보의 차이는?", "건물번호에 대해서 알려줘"],
    cache_examples=False,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch()

# Gradio 인터페이스 실행
# iface.launch(share=True)