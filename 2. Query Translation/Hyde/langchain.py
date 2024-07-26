from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings


directory = "Query Construction\VectorDBs\Documents"
loader = DirectoryLoader(directory, show_progress=True)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(documents=docs, 
                                    embedding=embeddings)

retriever = vectorstore.as_retriever()
llm = ChatOpenAI(api_key="api_key")

# Define a template for generating a scientific paper passage to answer a question
template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""
# Create a prompt object from the template for use in generating documents
prompt_hyde = ChatPromptTemplate.from_template(template)

# Define a pipeline to generate documents for retrieval using the HyDE prompt, OpenAI's chat model, and a string output parser
generate_docs_for_retrieval = (
    prompt_hyde | llm | StrOutputParser() 
)

# Define the question 
question = "What is task decomposition for LLM agents?"
# Generate documents by invoking the pipeline with the question
#generate_docs_for_retrieval.invoke({"question": question})

retrieval_chain = generate_docs_for_retrieval | retriever 
retireved_docs = retrieval_chain.invoke({"question": question})
print(retireved_docs)

# Define a template for answering a question based on the retrieved context
template = """Answer the following question based on this context:

{context}

Question: {question}
"""
# Create a prompt object from the template for use in the final response generation
prompt = ChatPromptTemplate.from_template(template)

# Define the final RAG (Retrieval-Augmented Generation) chain by linking the prompt, OpenAI's chat model, and a string output parser
final_rag_chain = (
    prompt
    | llm  # Assuming llm is a defined instance of a language model like ChatOpenAI
    | StrOutputParser()
)

# Generate the final response by invoking the RAG chain with the retrieved documents and the original question
response = final_rag_chain.invoke({"context": retireved_docs, "question": question})
print(response)