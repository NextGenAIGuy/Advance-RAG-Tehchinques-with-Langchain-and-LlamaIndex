# Import Packages
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from dotenv import loadenv
loadenv()

# Here i have loaded the text file from data directory using simpleDirectoryReader
documents_loader = SimpleDirectoryReader("data")
docs = documents_loader.load_data()

# Store the data into vectorStoreIndex
index = VectorStoreIndex.from_documents(docs)

# HyDEQueryTransform package will generate the hypothetical document then do similarity search from documents
query_engine = index.as_query_engine()
hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine, hyde)
query_str = "what did paul graham do after going to RISD"
response = hyde_query_engine.query(query_str)

print(response)
