from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from operator import itemgetter

# RAG-Fusion
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(api_key="api_key")
directory = "Query Construction\VectorDBs\Documents"
loader = DirectoryLoader(directory, show_progress=True)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(documents=docs, 
                                    embedding=embeddings)

retriever = vectorstore.as_retriever()

generate_queries = (
    prompt_rag_fusion 
    | ChatOpenAI(temperature=0)
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)


def reciprocal_rank_fusion(results: list[list], k=60):
    # Dictionary to store the fused scores of each unique document
    fused_scores = {}

    for docs in results:
        # Iterate through each document and its rank in the list
        for rank, doc in enumerate(docs):
            # Serialize the document to a string format to use as a dictionary key
            doc_str = dumps(doc)
            # Initialize the document score if it does not exist in the dictionary
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Update the document score using the Reciprocal Rank Fusion formula
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort documents by their fused scores in descending order
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked documents along with their fused scores
    return reranked_results

question = "What is Knowledge Graph"
retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
docs = retrieval_chain_rag_fusion.invoke({"question": question})
len(docs)


template = """Answer the following question based on this context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"question":question})