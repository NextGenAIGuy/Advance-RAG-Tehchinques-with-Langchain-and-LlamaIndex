#pip install chromadb
from langchain_community.vectorstores import Chroma

docs = "documents"
embeddings = "embedding model"
persist_directory = "path to the directory to store the vectorDB"

vectordb = Chroma.from_documents(documents=docs,
                      embedding=embeddings,
                      persist_directory=persist_directory)


## 1.How to delete the data from Chroma db?
ids = vectordb.get()['ids'] # Get all ids from vector db
vectordb.delete(ids=ids) # delete ids from vector db
vectordb.persist() # save the vector db to local disk


##2. How to remove all of the data from Chroma db?

vectordb.delete_collection()


##3. How to Provide the subset (Metadatas) in vector db so my result will be only from the subset not all the data from vector db?

# if you want to provide only one document as subset then use this code.
subset_list = [{'source': 'path1'}]
vectorstore_retriever = vectordb.as_retriever(
                    search_kwargs={
                        "k": 1,
                        "filter": subset_list
                    })


# In case of more than 1 docs as subset_list.
vectordb.as_retriever({
                        "k": 5,
                        "filter": {
                        "$or": subset_list
                            }
                        })
# In above code subset list should contain like this
[{'source': 'path1'},{'source':'path2'}]


## 3. How to get metadatas ?

vectordb.get()['metadatas']