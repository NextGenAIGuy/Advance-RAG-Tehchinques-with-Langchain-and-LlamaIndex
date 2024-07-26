from langchain.llms import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import SimpleVectorStore

# Initialize the LLM model from LangChain
llm = ChatOpenAI(temperature=0,api_key= "api_key")

# Simulated documents for each source
documents = {
    'web': ["Web Document 1: Latest trends in web development.", "Web Document 2: HTML, CSS, and JavaScript guide."],
    'database': ["Database Document 1: Relational vs NoSQL databases.", "Database Document 2: SQL optimization techniques."],
    'faq': ["FAQ Document 1: How to reset my password?", "FAQ Document 2: How to contact support?"]
}

# SimpleVectorStore for simulating document retrieval
vectorstore = {
    'web': SimpleVectorStore.from_texts(documents['web']),
    'database': SimpleVectorStore.from_texts(documents['database']),
    'faq': SimpleVectorStore.from_texts(documents['faq'])
}

def logical_routing(query):
    # Simple keyword-based routing
    if "web" in query:
        return 'web'
    elif "database" in query:
        return 'database'
    elif "faq" in query:
        return 'faq'
    else:
        return 'web'  # Default to web if no specific keyword is found

def retrieve_docs(source):
    # Retrieve documents from the chosen source
    return vectorstore[source].similarity_search("")

def generate_answer(query, retrieved_docs):
    # Create a combined input for the LLM
    combined_input = f"Query: {query}\n\nRetrieved Documents:\n" + "\n".join([doc.page_content for doc in retrieved_docs])
    # Generate the answer using the LLM
    answer = llm(combined_input)
    return answer

def answer_query(query):
    # Logical routing to determine the source
    source = logical_routing(query)
    # Retrieve documents from the selected source
    retrieved_docs = retrieve_docs(source)
    # Generate the final answer
    answer = generate_answer(query, retrieved_docs)
    return answer

# Example usage
query = "Tell me about web development."
answer = answer_query(query)
print(answer)
