from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader=WebBaseLoader('')
data=loader.load

text_splitter = RecursiveCharacterTextSplitter(chunck_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

ollama = Ollama(base_url='http://localhost:11434', model='llama3')

