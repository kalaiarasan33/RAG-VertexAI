from langchain_google_vertexai import VertexAI
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# ingestion steps 

# Step 1: load documents

# step 2: split documents into chunks

# step 3: embeding chunks to vector

# step 4 Store in vectordb

def ingestion():
    loader = PyPDFLoader("./rag-vertex/learn_ebpf_setup.pdf")
    documents = loader.load_and_split()

    embedding_model= VertexAIEmbeddings("textembedding-gecko")
    
    vectorstore = FAISS.from_documents(documents=documents,embedding=embedding_model)

    vectorstore.save_local("faiss_index_vectorstore")

#  Retriver

# Step 1: Embed user query 

# step 2: Semantic search in vector db

# step 3: Prompt augmentation

# step 4: Generation text
def retriever(message):
    llm = VertexAI(model_name="gemini-1.0-pro-002",temperature=0.3)
    embedding_model= VertexAIEmbeddings("textembedding-gecko")
    new_store = FAISS.load_local("faiss_index_vectorstore",embedding_model,allow_dangerous_deserialization=True)
    combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
    qa = create_retrieval_chain(new_store.as_retriever(),combine_docs_chain)
    res = qa.invoke(input={"input":message})
    print(res["answer"])

    
if __name__ =="__main__":
    ingestion()
    input_message = "explain how ebpf works with simple terms" 
    retriever(input_message)