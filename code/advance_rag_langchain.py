import os

os.environ["USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

import requests
# from langchain.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.load import dumps, loads

import operator

def prepare_data():
    loader = WebBaseLoader("https://wingfeitsang.github.io/home")
    documents = loader.load()
    txt_spliter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    chunks = txt_spliter.split_documents(documents) 
    return chunks

def embedding_data(chunks):
    # rag_embeddings = HuggingFaceBgeEmbeddings(model_name = "BAAI/bge-small-zh-v1.5")
    rag_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vect_store = Chroma.from_documents(documents = chunks, embedding = rag_embeddings, persist_directory = './chroma_langchain_db')
    retriever = vect_store.as_retriever()
    return vect_store, retriever


def retrieval_and_rank(queries):
    all_results = {}
    for query in queries:
        if query:
            search_results = vect_store.similarity_search_with_score(query)
            results = []
            for res in search_results:
                content = res[0].page_content
                score = res[1]
                results.append((content, score))
            all_results[query] = results
    
    document_ranks = []
    for query, doc_score_list in all_results.items():
        ranking_list = [doc for doc, _ in sorted(doc_score_list, key = lambda x: x[1], reverse=True)]
        document_ranks.append(ranking_list)
    return document_ranks

def reciprocal_rank_fusion(document_ranks, k=60):
    fused_scores = {}
    for docs in document_ranks:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1/(rank + k)
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x:x[1], reverse = True)
    ]
    return reranked_results

def get_multiple_queries(question):

    template = """You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple persepectives on the user question, your goals is to help the user overcome some of the limtations of the distance-based similarity search. Provide these alterntive questions separated by newlines. Original question: {question}
    """

    prompt_persepctives = ChatPromptTemplate.from_template(template)
    generate_queries = (
        prompt_persepctives
        | llm 
        | StrOutputParser()
        | (lambda x : x.split("\n"))
    )

    response = generate_queries.invoke({"question": question})
    print(response)

    all_results = retrieval_and_rank(response)
    reranked_results = reciprocal_rank_fusion(all_results)
    return generate_queries, reranked_results

def get_unique_union(documents: list[list]):
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]


def multi_query_generate_answer(question, generate_queries):
    template = """Answer the following question based this context:{context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    retrieval_chain = generate_queries | retriever.map() | get_unique_union
    final_rag_chain = (
        {"context": retrieval_chain,
         "question": operator.itemgetter("question")}
         | prompt
         | llm
         | StrOutputParser()
    )
    resp = final_rag_chain.invoke({"question": question})
    print(resp)


def generate_answer(question):
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | ChatPromptTemplate
        | llm 
        | StrOutputParser
    )
    resp = rag_chain.invoke(question)
    print(resp)


llm = OllamaLLM(model="deepseek-r1:8b")
template = """您是问答任务的助理。
请使用以下检索到的上下文来回答问题。
如果你不知道答案，就说不知道。
最多使用三句话，不超过100字来回答，保持答案简介。
Question: {question}
Context: {context}
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
chunks = prepare_data()

vect_store, retriever = embedding_data(chunks)
query = "我的名字叫做"
generate_queries, queries = get_multiple_queries(query)
multi_query_generate_answer(query, generate_queries)
