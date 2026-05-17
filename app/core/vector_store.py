from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os
from pathlib import Path

# Get the path to the current folder (app/core/)
current_dir = Path(__file__).parent
# Go up one level to the root where data_dictionary.txt lives
root_dir = current_dir.parent.parent
file_path = root_dir / "data_dictionary.txt"

# Use a small, fast local model for embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def setup_vector_db():
    if not file_path.exists():
        # This helps you debug exactly where the code is looking
        raise FileNotFoundError(f"Missing data_dictionary.txt at: {file_path.absolute()}")
        
    with open(file_path) as f:
        text = f.read()
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.create_documents([text])
    
    # Create local chroma DB
    vector_db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_db

# Create the retriever
vector_db = setup_vector_db()
retriever = vector_db.as_retriever(search_kwargs={"k": 1})