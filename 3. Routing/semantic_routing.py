from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Define prompts for different expertise areas
healthcare_prompt = """You are an experienced healthcare professional. 
You excel at providing detailed and accurate medical information and advice in a clear and understandable manner. 
When you don't know the answer to a question, you admit that you don't know.

Here is a question:
{query}"""

technology_prompt = """You are a knowledgeable technology expert. You have deep understanding of various technologies and are excellent at explaining technical concepts in a simple and engaging way. 
When you don't know the answer to a question, you admit that you don't know.

Here is a question:
{query}"""

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

prompt_texts = [healthcare_prompt, technology_prompt]
# Generate embeddings for the prompts
prompt_embeddings = embeddings.embed_documents(prompt_texts)

# Function to route the input query to the most appropriate prompt
def prompt_router(input):
    # Embed the input query
    query_embedding = embeddings.embed_query(input["query"])
    # Calculate similarity between query and each prompt
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    # Select the prompt with the highest similarity
    selected_prompt = prompt_texts[similarity.argmax()]
    # Log which prompt is chosen
    print("Using HEALTHCARE" if selected_prompt == healthcare_prompt else "Using TECHNOLOGY")
    return PromptTemplate.from_template(selected_prompt)

# Build the processing chain
chain = (
    {"query": RunnablePassthrough()}  
    | RunnableLambda(prompt_router) 
    | ChatOpenAI() 
    | StrOutputParser() 
)

# Test the chain with a sample query
print(chain.invoke("What are the latest advancements in AI technology?"))