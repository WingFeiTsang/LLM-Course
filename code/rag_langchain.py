from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.llms.base import LLM
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModel
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# loader = WebBaseLoader("https://baike.baidu.com/item/DeepSeek/65258669")
loader = WebBaseLoader("https://github.com/WingFeiTsang/LLM-Course/blob/main/SYLLABUS.md")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
splits = text_splitter.split_documents(docs)

# first download ollama and run "ollama run qwen2.5:0.5b"
embeddings = OllamaEmbeddings(model="qwen2.5:0.5b")
vector_store = Chroma.from_documents(documents = splits, embedding = embeddings)

retriever = vector_store.as_retriever()

# 考虑到下面代码本地没有GPU情况下运行DeepSeek R1时间比较长 本例子使用Ollama
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"  # 例如 "bert-base-uncased" 或 "gpt2" 等
# model = pipeline('text-generation', model=model_name)
# llm = HuggingFacePipeline(pipeline=model)
llm = Ollama(model="qwen2.5:7b")

prompt = hub.pull("rlm/rag-prompt")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("东北大学大语言模型课程的考核方式是？")
print(response)

# 生成的结果 考核方式包括课堂参与、完成指定任务以及实验。具体分数分配为：课堂参与5%，完成三项特定任务各5%，两项实验分别40%。
