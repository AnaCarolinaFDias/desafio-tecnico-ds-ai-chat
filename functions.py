from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.evaluation.qa import QAGenerateChain
from langchain_openai import ChatOpenAI
from datasets import Dataset
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import json
import random
from uuid import uuid4
import os
import json
import pandas as pd
from dotenv import load_dotenv
import getpass
import logging
import time

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
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
    
    embedding = OpenAIEmbeddings(model= model)

    vector_store = Chroma(
        collection_name="langchain_collection_OpenAI_embeddings",
        embedding_function=embedding,
        persist_directory="./langchain_collection", 
    )

    # Generate UUIDs for each document
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Process documents in batches of 1000
    for i in range(0, len(documents), 200):
        # logger.debug(f'Processing documents {i} to {min(i+1000, len(documents))}...')
        
        batch_documents = documents[i:i+200]
        batch_uuids = uuids[i:i+200]
        
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

def load_VectorStore(embeddings_provider: str) -> Chroma:
    """
    Loads the VectorStore based on the selected embeddings provider.

    Args:
        embeddings_provider (str): The embeddings provider to use. 
                                   It can be one of the following: 'HF', 'Google', 'OpenAI'.

    Returns:
        Chroma: The configured VectorStore object for persistence and embedding queries.

    Raises:
        ValueError: If the embeddings provider is unsupported.
        Exception: If an unexpected error occurs during the loading process.
    """
    try:
        # Select embeddings provider
        if embeddings_provider == 'HF':
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=os.getenv('HF_TOKEN'), 
                model_name="sentence-transformers/all-MiniLM-l6-v2"
            )
            directory = "./hf_collection"
            logger.info("HF embeddings loaded.")
        
        elif embeddings_provider == 'Google':
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            directory = "./google_collection"
            logger.info("Google embeddings loaded.")
        
        elif embeddings_provider == 'OpenAI':
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            directory = "./langchain_collection"
            logger.info("OpenAI embeddings loaded.")
        
        else:
            raise ValueError(f"Unsupported embeddings provider: {embeddings_provider}")
        
        # Check or create the directory
        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} does not exist. Creating it.")
            os.makedirs(directory)

        # Collection name
        collection_name = f"langchain_collection_{embeddings_provider}_embeddings"
        
        # Create VectorStore
        logger.info(f"Loading vector store for {collection_name}.")
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=directory
        )
        
        return vector_store

    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

def generate_and_save_validation_examples(docs, embeddings_provider, llm_model="gpt-4o-mini", select_n_documents=5):
    """
    Generate and save validation examples using a specified LLM and embeddings provider.

    Args:
        docs (list): A list of documents to generate validation examples from.
        embeddings_provider (str): The provider used for embeddings generation.
        llm_model (str, optional): The LLM model to use for QA generation. Defaults to "gpt-4o-mini".
        select_n_documents (int, optional): Number of random documents to select. Defaults to 5.

    Returns:
        list: A list of validation examples, where each example is a dictionary containing:
              - context (str): The original document.
              - query (str): The generated query.
              - ground_truths (str): The generated ground truth answers.
    """
    logger.debug(
        f"Creating {select_n_documents} validation examples with {llm_model} and {embeddings_provider}"
    )

    # Initialize the example generation chain
    example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))

    # Select `select_n_documents` random documents from the list `docs`
    random_documents = random.sample(docs, select_n_documents)

    # Format documents for input
    formatted_documents = [{"doc": doc} for doc in random_documents]

    # Apply the example generation chain and parse the results
    gen_examples = example_gen_chain.apply_and_parse(formatted_documents)

    # Adjust the output to include context
    gen_examples_adjusted = []
    for doc, example in zip(random_documents, gen_examples):
        qa_pair = example.get("qa_pairs", {})
        query = qa_pair.get("query", "")
        answer = qa_pair.get("answer", "")

        gen_examples_adjusted.append({
            "context": doc,  # Add the original document as context
            "query": query,
            "ground_truths": answer
        })

    validation_examples = gen_examples_adjusted

    # Prepare the results in the expected format
    documents_dict = [
        [
            {
                "context": doc["context"].page_content,
                "metadata": doc["context"].metadata,
                "ground_truths": doc["ground_truths"],
                "query": doc["query"]
            }
            for doc in data
        ]
        for data in [validation_examples]
    ]

    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)

    # Save the results to a JSON file
    results_filename = f"results/query_{embeddings_provider}_results.json"
    with open(results_filename, "w") as file:
        json.dump(documents_dict, file, indent=4)

    logger.debug(f"Documents have been saved to {results_filename}")

    return validation_examples


def baseline_similarity_method(validation_examples, vector_store, embeddings_provider):
    """
    Implements a baseline retrieval-augmented generation (RAG) method to answer queries 
    using a similarity-based retriever and a language model.

    Args:
        validation_examples (list): A list of validation examples to query the model.
        vector_store (object): A vector store instance for similarity-based retrieval.
        embeddings_provider (str): The name of the embeddings provider (used in result file naming).

    Returns:
        list: A dictionary containing the retrieved contexts, metadata, queries, 
              ground truths, and generated answers for each validation example.

    """
    logger.debug("Baseline similarity method to answer queries initiated.")
    
    start_time = time.time()
    model = 'gpt-3.5-turbo-instruct'

    # Initialize the language model
    llm = OpenAI(temperature=0.2, model=model)

    # Set up the retriever with similarity search
    retriever = vector_store.as_retriever(search_type="similarity", k=5)  # Retrieve the most relevant chunks

    # Initialize the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True,
        chain_type_kwargs={"document_separator": "\n"}
    )

    # Apply the model to the validation examples
    predictions = qa.apply(validation_examples)

    # Prepare the results
    baselinemethod_documents_dict = [
        [
            {
                "context": doc['context'].page_content,
                "metadata": doc['context'].metadata,
                "query": doc['query'],
                "ground_truths": doc['ground_truths'],
                "answer": doc['result']
            }
            for doc in data
        ]
        for data in [predictions]
    ]

    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)

    # Save the results to a JSON file
    results_filename = f"results/query_{embeddings_provider}_baselinemethod_documents_dict_results.json"
    with open(results_filename, "w") as file:
        json.dump(baselinemethod_documents_dict, file, indent=4)

    print(f"Documents have been saved to '{results_filename}'")
    
    end_time = time.time()
    logger.debug("Baseline similarity method documents have been saved to %s", results_filename)

    Result_time = end_time - start_time

    return Result_time, baselinemethod_documents_dict

def contextualize_query_with_categories(vector_store, validation_examples, embeddings_provider):
    """
    Enhances queries by contextualizing them with relevant categories before performing a document search and generating answers.

    Args:
        vector_store (object): The vector store used for similarity-based retrieval of documents.
        validation_examples (list): A list of validation examples containing queries and expected ground truths.
        embeddings_provider (str): The name of the embeddings provider, used in naming the results file.

    Returns:
        list: A dictionary containing the contextualized queries, retrieved contexts, metadata, ground truths, 
              and generated answers for each validation example.
    """
    logger.debug("Contextualized query method to answer queries initiated.")

    start_time = time.time()

    # Define the high-level query template for category search
    category_prompt = """
    You are an expert in outdoor clothing products. Based on the following query, 
    return a list of relevant categories (e.g., jackets, pants) from the catalog.
    Query: {query}
    """
    category_template = PromptTemplate(input_variables=["query"], template=category_prompt)
    category_chain = LLMChain(prompt=category_template, llm=OpenAI())

    # Define the refined search for documents within each category
    document_prompt = """
    You are an assistant for question-answering tasks about the products from a store that sells outdoor clothing. 
    Your function is to use the catalog of retrieved context to answer the client's questions based on the following query and the category {category}.
    Query: {query}
    Category: {category}
    Catalog: {catalog}
    """
    document_template = PromptTemplate(input_variables=["query", "category", "catalog"], template=document_prompt)
    document_chain = LLMChain(prompt=document_template, llm=OpenAI())

    for index, item in enumerate(validation_examples):
        query = item['query']
        
        # Obtain the relevant category for the query
        category_result = category_chain.run({"query": query})
        logger.debug("Category identified for query '%s': %s", query, category_result)

        # Combine query and category for refined search
        modified_query = f"{query} Category: {category_result}"
        catalog = vector_store.similarity_search(modified_query, k=5)  # Retrieve relevant catalog entries

        # Generate refined answer using retrieved catalog
        refined_answer = document_chain.run({
            "query": query,
            "category": category_result,
            "catalog": catalog
        })
        validation_examples[index]['result'] = refined_answer

    # Prepare the results in dictionary format
    contextualize_query_with_categories_dict = [
        [
            {
                "context": doc['context'].page_content,
                "metadata": doc['context'].metadata,
                "query": doc['query'],
                "ground_truths": doc['ground_truths'],
                "answer": doc['result'],
            }
            for doc in data
        ]
        for data in [validation_examples]
    ]

    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)

    # Save the results to a JSON file
    results_filename = f"results/query_{embeddings_provider}_contextualize_query_with_categories_dict_results.json"
    with open(results_filename, "w") as file:
        json.dump(contextualize_query_with_categories_dict, file, indent=4)

    print(f"Documents have been saved to '{results_filename}'")
    logger.debug("Contextualized query method documents have been saved to %s", results_filename)
    
    end_time = time.time()
    Result_time = end_time - start_time

    return Result_time, contextualize_query_with_categories_dict

def hierarchical_retrieval_with_categories(vector_store, validation_examples, embeddings_provider):
    """
    Implements a hierarchical retrieval method to answer queries by categorizing and refining the search process.

    This method uses a two-step approach:
    1. Identifies relevant categories based on the input query using a language model.
    2. Retrieves relevant documents within each identified category and refines the answer using another language model.

    Parameters:
        vector_store (VectorStore): The vector store for similarity-based document retrieval.
        validation_examples (list): List of validation examples, each containing a query and related information.
        embeddings_provider (str): The name of the embeddings provider used for vectorizing the documents.

    Returns:
        list: A list of dictionaries containing context, metadata, query, ground truths, and answers.
    """
    logger.debug("Hierarchical retrieval method to answer queries initiated")

    start_time = time.time()

    # Define the high-level query template for category search
    category_prompt = """
    You are an expert in outdoor clothing products. Based on the following query, 
    return a list of relevant categories (e.g., jackets, pants) from the catalog.
    Query: {query}
    """
    category_template = PromptTemplate(input_variables=["query"], template=category_prompt)
    category_chain = LLMChain(prompt=category_template, llm=OpenAI())

    # Define the refined search for documents within each category
    document_prompt = """
    You are an assistant for question-answering tasks about the products from a store that sells outdoor clothing. 
    Your function is to use the catalog of retrieved context to answer the client's questions based on the following query.
    Query: {query}
    Catalog: {catalog}
    """
    document_template = PromptTemplate(input_variables=["query", "catalog"], template=document_prompt)
    document_chain = LLMChain(prompt=document_template, llm=OpenAI())

    for index, item in enumerate(validation_examples):
        query = item['query']
        category_result = category_chain.run({"query": query})
    
        # Convert category_result to vector (embedding) using the same model as the vector store
        relevant_documents = vector_store.similarity_search(category_result, k=5)

        if not relevant_documents:
            logger.debug(f"No relevant documents found for category '{category_result}'")  
            continue
        
        # Refine the answer using the filtered documents
        refined_answer = document_chain.run({"query": query, "catalog": relevant_documents})
        validation_examples[index]['result'] = refined_answer

    hierarchicalmethod_documents_dict = [
        [
            {
                "context": doc['context'].page_content, 
                "metadata": doc['context'].metadata,
                "query": doc['query'],
                "ground_truths": doc['ground_truths'],  
                "answer": doc['result'],
            }
            for doc in data
        ]
        for data in [validation_examples]
    ]

    os.makedirs('results', exist_ok=True)

    # Save the list of dictionaries to a JSON file
    results_filename = f"results/query_{embeddings_provider}_hierarchicalmethod_documents_dict_results.json"
    with open(results_filename, "w") as file:
        json.dump(hierarchicalmethod_documents_dict, file, indent=4)
        
    print(f"Documents have been saved to '{results_filename}'")
    logger.debug(f"Hierarchical method Documents have been saved to {results_filename}")

    end_time = time.time()
    Result_time = end_time - start_time
    
    return Result_time, hierarchicalmethod_documents_dict




def validate_results(file_path):
    """
    Validate and transform results from a JSON file into a structured Dataset.

    Args:
        file_path (str): Path to the JSON file containing the results.

    Returns:
        Dataset: A Hugging Face Dataset object with the structured results.
    """
    # Read the JSON file
    with open(file_path, "r") as file:
        content = file.read()  # Read the entire content as a string
        baseline_results = json.loads(content)

    # Flatten the list of results and rename it to `database_result`
    database_result = [item for sublist in baseline_results for item in sublist]

    # Initialize lists to store the values
    queries = []        # List for queries
    answers = []        # List for predicted answers
    contexts = []       # List for context
    ground_truths = []  # List for ground truth answers

    # Populate the lists
    for result in database_result:
        queries.append(result.get('query', ''))
        answers.append(result.get('answer', ''))
        contexts.append([result.get('context', '')])
        ground_truths.append(result.get('answer', ''))

    # Create the final dictionary
    final_dict = {
        "question": queries,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    # Convert the dictionary to a Dataset
    dataset_results = Dataset.from_dict(final_dict)

    return dataset_results

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


def compare_time_methods(validation_examples, vector_store, embeddings_provider,baseline_similarity_method, contextualize_query_with_categories, hierarchical_retrieval_with_categories ):
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
    baseline_time = evaluate_time(baseline_similarity_method, validation_examples, vector_store, embeddings_provider)
    contextualize_time = evaluate_time(contextualize_query_with_categories,validation_examples, vector_store, embeddings_provider)
    hierarchical_time = evaluate_time(hierarchical_retrieval_with_categories,validation_examples, vector_store, embeddings_provider)

    # Final comparison
    comparison = {
        'Method': ['Baseline', 'Contextualize', 'Hierarchical'],
        'Time Taken (s)': [baseline_time, contextualize_time, hierarchical_time],
    }
    
    return comparison
