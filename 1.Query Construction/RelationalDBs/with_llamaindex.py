# Import packages
from sqlalchemy import create_engine
from llama_index.core import SQLDatabase
from sqlalchemy import text
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.openai import OpenAI


# Create engine and database
engine = create_engine("sqlite:///example.db")
sql_database = SQLDatabase(engine, include_tables=["employees"])

# Test is it working
with engine.connect() as con:
    rows = con.execute(text("SELECT * from employees"))
    for row in rows:
        print(row)



# Initiate llm model and we have use NLSQLTableQuery Engine which will convert Natural language to Structured Query language.
llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database, tables=["employees"], llm=llm
)
query_str = "How many employees are there?"
response = query_engine.query(query_str)
print(response)