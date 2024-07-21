## In this section we convert text to sql query using create_sql_query_chain from langchain
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase

import os
db = SQLDatabase.from_uri("sqlite:///example.db")


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,api_key="api_key")
chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "How many employees are there"})
result = db.run(response)
print(result)