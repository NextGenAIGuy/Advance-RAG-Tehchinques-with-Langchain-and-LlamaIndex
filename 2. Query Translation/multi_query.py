#import Packages
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


## initiate Vectordb here i have used persist vectordb which means i am loading the locally stored vectordb.
vectordb_path = "Vector_db"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=vectordb_path,
                                       embedding_function=embeddings)
vectorstore_retriever = vectordb.as_retriever(
                search_kwargs={
                    "k": 2
                }
            )
llm = ChatOpenAI(temperature=0,api_key="api_key")

## Here we use MultiQueryRetriever from langcahin which will generate multiple queriesand retriever data accordingly.
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore_retriever, llm=llm
)

question = "How to store the data into VectorDB?"

unique_docs = retriever_from_llm.invoke(question)
len(unique_docs)

## Output
output = ['What are the best practices for indexing and storing data in a vector database?', 
 'How can I optimize my data storage for efficient querying in a vector-based database?'
, 'What are the key considerations for designing a scalable and efficient data storage system using vector databases?']