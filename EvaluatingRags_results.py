import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_similarity, answer_correctness, context_precision, context_recall, context_entity_recall

from functions import validate_results

# Example usage:
file_path_baseline = "results/query_OpenAI_baselinemethod_documents_dict_results.json"
RAG_baseline_results = validate_results(file_path_baseline)

file_path_contextualized = "results/query_OpenAI_contextualize_query_with_categories_dict_results.json"
RAG_contextualized_results = validate_results(file_path_baseline)

file_path_hierarchical = "results/query_OpenAI_hierarchicalmethod_documents_dict_results.json"
RAG_hierarchical_results = validate_results(file_path_hierarchical)



RAG_baseline_results_metrics = evaluate(RAG_baseline_results, metrics=[faithfulness, answer_correctness, answer_relevancy,
                                       context_precision, context_recall, context_entity_recall, 
                                       answer_similarity, answer_correctness])

RAG_baseline_results_df = RAG_baseline_results_metrics.to_pandas()
RAG_baseline_results_df['method']= 'baseline'

RAG_contextualized_results_metrics = evaluate(RAG_contextualized_results, metrics=[faithfulness, answer_correctness, answer_relevancy,
                                       context_precision, context_recall, context_entity_recall, 
                                       answer_similarity, answer_correctness])

RAG_contextualized_results_df = RAG_contextualized_results_metrics.to_pandas()
RAG_contextualized_results_df['method']= 'Contextualized'


RAG_hierarchical_results_metrics = evaluate(RAG_hierarchical_results, metrics=[faithfulness, answer_correctness, answer_relevancy,
                                       context_precision, context_recall, context_entity_recall, 
                                       answer_similarity, answer_correctness])

RAG_hierarchical_results_df = RAG_hierarchical_results_metrics.to_pandas()
RAG_hierarchical_results_df['method']= 'hierarchical'

comparing_results = pd.concat([RAG_baseline_results_df, RAG_contextualized_results_df, RAG_hierarchical_results_df],axis=0)

# Select only numeric columns
numeric_comparing_results = pd.concat([comparing_results.select_dtypes(include='number'), comparing_results['method']], axis=1)

mean_result_by_method = numeric_comparing_results.groupby(['method'])\
                                                .agg('mean').reset_index()

# Save the results to a JSON file
results_filename = f"results/comparing_methods_meanresults.csv"

# Save the DataFrame to a CSV file
mean_result_by_method.to_csv(results_filename, index=False)  # `index=False` prevents writing row numbers

print("Documents have been saved to {results_filename}")

numeric_comparing_results = pd.concat([comparing_results.select_dtypes(include='number'), comparing_results['user_input']], axis=1)

mean_result_by_method_userinput = numeric_comparing_results.groupby(['user_input'])\
                                                           .agg('mean').reset_index()   

# Save the results to a JSON file
results_filename = f"results/comparing_methods_meanresults_byinputs.csv"

# Save the DataFrame to a CSV file
mean_result_by_method_userinput.to_csv(results_filename, index=False)  # `index=False` prevents writing row numbers

print("Documents have been saved to {results_filename}")