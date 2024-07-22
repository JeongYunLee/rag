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
import csv


# .env 파일 활성화
load_dotenv()

from kiwipiepy import Kiwi

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

############################################################################################################
######### 파일 불러오기 #######################################################################################
############################################################################################################

loader = DirectoryLoader(path='data/', glob='*.txt', loader_cls=TextLoader, show_progress=True, use_multithreading=True)
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

# kiwi = Kiwi()

# def kiwi_tokenize(text):
#     return [token.form for token in kiwi.tokenize(text)]


# BM25 retrieval 
bm25_retriever = BM25Retriever.from_documents(child_docs) #preprocess_func = kiwi_tokenize
bm25_retriever.k = 1  # BM25Retriever의 검색 결과 개수를 1로 설정합니다.

#DB retrieval 
embedding = OpenAIEmbeddings(model='text-embedding-3-large')  # OpenAI 임베딩을 사용합니다.

vectorstore = Chroma(client=client, collection_name="ten_files_openai_3072", embedding_function=embedding)
chromadb_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Ensemble retrieval 
ensemble_retriever = EnsembleRetriever(
    retrievers=[chromadb_retriever, bm25_retriever],
    weights=[0.5, 0.5],
    search_type="mmr",
)

############################################################################################################
############# Gradio 인터페이스 생성 #############################################################################
############################################################################################################

def chat(query, history):
    print(query)
    ensemble_result = ensemble_retriever.invoke(query) 
    print(ensemble_result)
    max_similarity_doc = ensemble_result[0].page_content + '\n######\n' + ensemble_result[1].page_content

    chat = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)


    messages = [
        SystemMessage(content=f"""
                    너는 Document의 정보를 반드시 활용해서 답변을 생성하는 챗봇이야. 
                    이때, 답변은 Document에 정보가 있을 수도 있고, 없을 수도 있어. 
                    Document의 정보로 답변을 생성할 수 있는 경우 해당 정보를 활용하고, 만약 Document의 정보로 답변을 유추조차 할 수 없는 경우, 새로운 정보를 생성하지 말고 '정보가 부족해서 답을 할 수 없습니다' 라고 답변해줘.
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

    return output.content, max_similarity_doc

def handle_submit(query, history):
    response, example = chat(query, history)
    history.append((query, response))
    
    # Log the query, response, and example output to log.csv
    log_to_csv([query, response, example])
    
    return "", history, example

def clear_fields():
    return "", [], ""

like_status = None  # Define like_status globally

def vote(data):
    global like_status  # Declare like_status as global to modify it within the function
    
    if data and hasattr(data, 'liked'):
        if data.liked:
            like_status = "like"
        else:
            like_status = "dislike"
    else:
        like_status = ""  # If data is None or doesn't have 'liked' attribute, set like_status to empty string

    # Log the feedback to CSV
    log_to_csv([like_status])

def log_to_csv(data):
    log_file = "flagged/log_blocks.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Query", "Response", "Example Output", "Feedback"])
        writer.writerow(data)

with gr.Blocks(theme=gr.themes.Soft()) as demo:


    chatbot = gr.Chatbot(label="Chatbot", height=800, show_copy_button=True)
    chatbot.like(vote, None, None)
    query = gr.Textbox(label="Query")
    with gr.Accordion("Resource", open=False):
        example_output = gr.Textbox(label="", lines=20)  

    query.submit(handle_submit, [query, chatbot], [query, chatbot, example_output])

    with gr.Row():
        btn = gr.Button("Generate")
        btn.click(handle_submit, inputs=[query, chatbot], outputs=[query, chatbot, example_output])

        clear_btn = gr.Button("Clear")
        clear_btn.click(clear_fields, inputs=[], outputs=[query, chatbot, example_output])

    gr.Examples(["주소와 주소정보의 차이점을 알려줘", "사물주소란 무엇일까?"], inputs=[query])

demo.launch()