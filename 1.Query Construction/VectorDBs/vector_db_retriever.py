# Install the required packages
#pip install langchain, langchain-community, sentece_transformers
#pip install chromadb
#pip install unstructured[pdf], unstructured[doc].

# Import libraries
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_groq import ChatGroq


directory = "Query Construction\VectorDBs\Documents"
loader = DirectoryLoader(directory, show_progress=True)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

persist_directory = "local_path"
vectordb = Chroma.from_documents(documents=docs,
                      embedding=embeddings,
                      persist_directory=persist_directory)

# If you want to use the Faiss Vector DB
'''pip install faiss-cpu
from langchain_community.vectorstores import FAISS
db = FAISS.from_documents(documents, Embeddings)'''


vectorstore_retriever = vectordb.as_retriever(
                    search_kwargs={
                        "k": 1 })
#retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})
#retriever = db.as_retriever(search_type="mmr")



turbo_llm = HuggingFaceHub(repo_id="google/flan-t5-large",  model_kwargs={"temperature":0.5, "max_length":512},
                                                 huggingfacehub_api_token="huggingface_api")

'''turbo_llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key= "api_key"
)'''
qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm,
                                          chain_type="stuff",
                                          retriever=vectorstore_retriever,
                                          return_source_documents=True)

warning = "Please refrain from speculating if you're unsure. Simply state that you don't know. Answers should be concise, within 200 words."
question = warning + "You are a helpfull AI Assistant. Your Job is to generate output based on the query."
Query = input("Please Enter your Query")
query = question + " Requirement: " +  Query
llm_response = qa_chain(query)
print(llm_response)