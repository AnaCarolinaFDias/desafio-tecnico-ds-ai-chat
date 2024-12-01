import tqdm as notebook_tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from uuid import uuid4
import os
import json
import pandas as pd
from dotenv import load_dotenv
import getpass
import logging

# Configurar o logging para salvar a saída de debug em um arquivo
logging.basicConfig(
    filename='./logs/debug_output.log',  # O arquivo onde os logs serão salvos
    level=logging.DEBUG,          # O nível de log (DEBUG para capturar tudo)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Formato do log
)

logger = logging.getLogger()
load_dotenv()

def openai_embedding_function(documents, model = "text-embedding-3-large"):
    """
    Function to process and add documents in batches to a vector store with generated UUIDs.

    Parameters:
    - documents: List of documents to be added to the vector store.

    Returns:
    None
    """
    # Variable initialization
    os.getenv("OPENAI_API_KEY")
    
    embedding = OpenAIEmbeddings(model= model)

    vector_store = Chroma(
        collection_name="langchain_collection_OpenAI_embeddings",
        embedding_function=embedding,
        persist_directory="./langchain_collection", 
    )

    # Generate UUIDs for each document
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Process documents in batches of 1000
    for i in range(0, len(documents), 1000):
        # logger.debug(f'Processing documents {i} to {min(i+1000, len(documents))}...')
        
        batch_documents = documents[i:i+1000]
        batch_uuids = uuids[i:i+1000]
        
        # Log for each batch
        logger.debug(f'Number of documents in the batch: {len(batch_documents)}')
        
        # Add documents to the vector store
        try:
            vector_store.add_documents(documents=batch_documents, ids=batch_uuids)
            logger.info(f'Batch of documents {i} added successfully.')
        except Exception as e:
            logger.error(f'Error adding documents in batch {i}: {e}')

    logger.debug("Processing completed - OPENAI embeddings generated.")

def hf_embeddings_function(documents, model= 'all-MiniLM-l6-v2'):
    """
    Function to process and add documents in batches to a vector store with generated UUIDs.

    Parameters:
    - documents: List of documents to be added to the vector store.

    Returns:
    None
    """
    
    if not os.getenv("HF_TOKEN"):
           os.environ["HF_TOKEN"] = getpass.getpass("Enter your HF_TOKEN: ")

    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv('HF_TOKEN'), model_name= f"sentence-transformers/{model}"
    )

    # Initialize the vector store (Chroma)
    vector_store = Chroma(
        collection_name="langchain_collection_HF_embeddings",
        embedding_function=embeddings,
        persist_directory="./hf_collection", 
    )

    # Generate UUIDs for each document
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Add documents in batches of 1000
    for i in range(0, len(documents), 1000):
        # logger.debug(f'Processing documents {i} to {min(i+1000, len(documents))}...')
        
        batch_documents = documents[i:i+1000]
        batch_uuids = uuids[i:i+1000]
        
        # Log for each batch
        logger.debug(f'Number of documents in the batch: {len(batch_documents)}')
        
        # Add documents to the vector store
        try:
            vector_store.add_documents(documents=batch_documents, ids=batch_uuids)
            logger.info(f'Batch of documents {i} added successfully.')
        except Exception as e:
            logger.error(f'Error adding documents in batch {i}: {e}')

    logger.debug("Processing completed - HuggingFace embeddings generated.")

def google_embedding_function(documents, model = "text-embedding-004"):
    """
    Function to process and add documents in batches to a vector store with generated UUIDs.

    Parameters:
    - documents: List of documents to be added to the vector store.

    Returns:
    None
    """
    # Variable initialization
    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your GOOGLE_API_KEY: ")
   
    embeddings = GoogleGenerativeAIEmbeddings(model=f"models/{model}")

    vector_store = Chroma(
    collection_name="langchain_collection_Google_embeddings",
    embedding_function=embeddings,
    persist_directory="./google_collection", 
    )
    
    # Generate UUIDs for each document
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Process documents in batches of 1000
    for i in range(0, len(documents), 1000):
        # logger.debug(f'Processing documents {i} to {min(i+1000, len(documents))}...')
        
        batch_documents = documents[i:i+1000]
        batch_uuids = uuids[i:i+1000]
        
        # Log for each batch
        logger.debug(f'Number of documents in the batch: {len(batch_documents)}')
        
        # Add documents to the vector store
        try:
            vector_store.add_documents(documents=batch_documents, ids=batch_uuids)
            logger.info(f'Batch of documents {i} added successfully.')
        except Exception as e:
            logger.error(f'Error adding documents in batch {i}: {e}')

    logger.debug("Processing completed - google embeddings generated.")

def load_VectorStore(embeddings_provider):
    try:
        if embeddings_provider == 'HF':
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=os.getenv('HF_TOKEN'), model_name="sentence-transformers/all-MiniLM-l6-v2"
            )
            directory = "./hf_collection"
            print("HF db loaded")
        
        elif embeddings_provider == 'Google':
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            directory = "./google_collection"
            print("Google db loaded")
        
        elif embeddings_provider == 'OpenAI':
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            directory = "./langchain_collection"
            print("OpenAI db loaded")
        
        else:
            raise ValueError(f"Unsupported embeddings provider: {embeddings_provider}")
        
        collection_name = f"langchain_collection_{embeddings_provider}_embeddings"
        
        # Verifica se o diretório existe e avisa ao usuário caso não exista
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist. Please create the directory before proceeding.")
        
        # Cria o vector store
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=directory
        )  

        logger.debug(f"Loading vector store from {collection_name}")
        
        return vector_store

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
def measure_time(func, *args, **kwargs):
    """
    Measure the execution time of a given function.
    
    Args:
        func (callable): The function to measure.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        tuple: A tuple containing the function result and the time taken (in seconds).
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def evaluate_time(func, *args, **kwargs):
    """
    Measure the execution time of a function without returning its result.

    Args:
        func (callable): The function to measure.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        float: The time taken by the function (in seconds).
    """
    _, time_taken = measure_time(func, *args, **kwargs)
    return time_taken


def compare_time_methods(validation_examples, vector_store, embeddings_provider):
    """
    Compare the performance of baseline and generative retrieval methods.

    Args:
        query (str): The user query.
        vector_store (object): The vector store for similarity search.
        expected_results (list): The expected results for precision evaluation.
        n_chunks (int, optional): The number of chunks to retrieve for the baseline method. Defaults to 5.

    Returns:
        dict: A dictionary containing the comparison of time taken and precision for both methods.
    """
    # Baseline method evaluation
    baseline_time = evaluate_time(baseline_retrieval, validation_examples, vector_store, embeddings_provider)
    contextualize_time = evaluate_time(contextualize_query_with_categories,validation_examples, vector_store, embeddings_provider)
    hierarchical_time = evaluate_time(hierarchical_retrieval_with_categories,validation_examples, vector_store, embeddings_provider)

    # Final comparison
    comparison = {
        'Method': ['Baseline', 'Contextualize', 'Hierarchical'],
        'Time Taken (s)': [baseline_time, contextualize_time, hierarchical_time],
    }
    
    return comparison
