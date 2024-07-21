## Here we have used Groq llama-3 model which is free. We can directly generate the sql from text with llm
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv("groq_api_key")

db = SQLDatabase.from_uri("sqlite:///example.db")

# Initiat LLM model 
llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key= api_key
)
system = "You are a helpful assistant. Your job is to convert the text to SQL. Only give the SQL query"
human = "{question}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | llm
response = chain.invoke({"question": "How many employees are there"})
print(response.content)

result = db.run(response.content)
print(result)