import tqdm as notebook_tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from uuid import uuid4
import os
import json
import pandas as pd
from dotenv import load_dotenv
import logging

# Configurar o logging para salvar a saída de debug em um arquivo
logging.basicConfig(
    filename='./logs/debug_output.log',  # O arquivo onde os logs serão salvos
    level=logging.DEBUG,          # O nível de log (DEBUG para capturar tudo)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Formato do log
)

logger = logging.getLogger()
load_dotenv()

def openai_embedding_function(documents):
    """
    Function to process and add documents in batches to a vector store with generated UUIDs.

    Parameters:
    - documents: List of documents to be added to the vector store.

    Returns:
    None
    """
    # Variable initialization
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store = Chroma(
        collection_name="langchain_collection_OpenAI_embeddings",
        embedding_function=embedding,
        persist_directory="./langchain_collection", 
    )

    # Generate UUIDs for each document
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Process documents in batches of 1000
    for i in range(0, len(documents), 1000):
        logger.debug(f'Processing documents {i} to {min(i+1000, len(documents))}...')
        
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

    logger.info("Processing completed.")    

def hf_embeddings_function(documents):
    """
    Function to process and add documents in batches to a vector store with generated UUIDs.

    Parameters:
    - documents: List of documents to be added to the vector store.

    Returns:
    None
    """

    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv('HF_TOKEN'), model_name="sentence-transformers/all-MiniLM-l6-v2"
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
        logger.debug(f'Processing documents {i} to {min(i+1000, len(documents))}...')
        
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

    logger.info("Processing completed.")
    

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


def baseline_retrieval(query, vector_store, n=5):
    """
    Perform baseline retrieval of the top-n most similar chunks.

    Args:
        query (str): The user query.
        vector_store (object): The vector store for similarity search.
        n (int, optional): The number of top similar chunks to retrieve. Defaults to 5.

    Returns:
        list: A list of the top-n retrieved documents.
    """
    retrieval_results = vector_store.similarity_search(query, k=n)
    return retrieval_results


def generative_retrieval(query, vector_store, model='gpt-3.5-turbo'):
    """
    Perform retrieval-augmented generation using a generative model.

    Args:
        query (str): The user query.
        vector_store (object): The vector store for similarity search.
        model (str, optional): The model name for the generative task. Defaults to 'gpt-3.5-turbo'.

    Returns:
        str: The generated response based on the retrieved documents.
    """
    retrieval_results = baseline_retrieval(query, vector_store)
    llm = OpenAI(temperature=0, model=model)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    response = chain.run(query)
    return response


def evaluate_precision(results, expected_results):
    """
    Evaluate the precision of the retrieved results.

    Args:
        results (list): The retrieved results.
        expected_results (list): The expected results for evaluation.

    Returns:
        float: The precision score as a value between 0 and 1.
    """
    precision = np.mean([1 if result in expected_results else 0 for result in results])
    return precision


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


def compare_methods(query, vector_store, expected_results, n_chunks=5):
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
    baseline_results = baseline_retrieval(query, vector_store, n=n_chunks)
    baseline_time = evaluate_time(baseline_retrieval, query, vector_store, n=n_chunks)
    baseline_precision = evaluate_precision(baseline_results, expected_results)

    # Generative method evaluation
    generative_results = generative_retrieval(query, vector_store)
    generative_time = evaluate_time(generative_retrieval, query, vector_store)
    generative_precision = evaluate_precision([generative_results], expected_results)

    # Final comparison
    comparison = {
        'Method': ['Baseline', 'Generative'],
        'Time Taken (s)': [baseline_time, generative_time],
        'Precision': [baseline_precision, generative_precision],
    }
    
    return comparison
