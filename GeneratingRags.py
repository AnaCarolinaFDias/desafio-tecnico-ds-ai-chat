import os
from dotenv import load_dotenv
import getpass
import logging
import pandas as pd
import json
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_similarity, answer_correctness, context_precision, context_recall, context_entity_recall
from functions import load_VectorStore, generate_and_save_validation_examples, baseline_similarity_method, contextualize_query_with_categories, hierarchical_retrieval_with_categories


# Configurar o logging para salvar a saída de debug em um arquivo
logging.basicConfig(
    filename='./logs/debug_output.log',  # O arquivo onde os logs serão salvos
    level=logging.DEBUG,          # O nível de log (DEBUG para capturar tudo)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Formato do log
)

logger = logging.getLogger()

import warnings
warnings.filterwarnings("ignore")

load_dotenv() # read local .env file

if not os.getenv("OPENAI_API_KEY"):
   os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

file = 'inputs/OutdoorClothingCatalog_1000_withCategories.csv'
loader = CSVLoader(file_path=file, encoding="utf-8")
docs = loader.load()

text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=10)

split_documents = []
for doc in docs:
    split_docs = text_splitter.split_documents([doc])
    split_documents.extend(split_docs)
    
documents_dict = [
    {"page_content": doc.page_content, "metadata": doc.metadata} for doc in split_documents
]
# Save the list of dictionaries to a JSON file
with open("inputs/documents_split_langchain.json", "w") as file:
    json.dump(documents_dict, file, indent=4)

logger.debug("Documents have been saved to 'documents_split_langchain.json'")

# Load documents split
with open("inputs/documents_split_langchain.json", "r") as file:
    documents_dict = json.load(file)

# Convert the list of dictionaries back to a list of Document objects
documents = [
    Document(page_content=doc["page_content"], metadata=doc["metadata"])
    for doc in documents_dict
]

# Generating embeddings 
# openai_embedding_function(documents, model = "text-embedding-3-large")

embeddings_provider = 'OpenAI'
vector_store = load_VectorStore(embeddings_provider)

validation_examples = generate_and_save_validation_examples(docs, embeddings_provider, llm_model="gpt-4o-mini", select_n_documents=5)

baseline_time, baselinemethod_documents_dict = baseline_similarity_method(validation_examples, vector_store, embeddings_provider)
print("baseline_similarity_method Documents have been saved.")

contextualize_time, contextualize_query_with_categories_dict = contextualize_query_with_categories(vector_store, validation_examples,embeddings_provider)
print("Contextualized query method Documents have been saved.")

hierarchical_time, hierarchicalmethod_documents_dict = hierarchical_retrieval_with_categories(vector_store, validation_examples,embeddings_provider)
print("Hierarchical method Documents have been saved.")


# Final comparison
comparison = {
    'Method': ['Baseline', 'Contextualize', 'Hierarchical'],
    'Time Taken (s)': [baseline_time, contextualize_time, hierarchical_time],
}
    
results_filename = f"results/RAG_Methods_time.json"
with open(results_filename, "w") as file:
    json.dump(comparison, file, indent=4)
