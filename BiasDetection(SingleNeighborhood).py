#!/usr/bin/env python
# coding: utf-8

# In[1]:
import warnings

import pandas as pd
from tqdm import tqdm

from Argumentation import construct_graph, construct_graph_QE
import utility

warnings.filterwarnings("ignore")
import random
from datasets import load_data, dataset_types, local_synthetic, datasets
from models import load_logisitc_regression, model_types
from neighborhood import neighbourhood_selector, neighbourhoods, measure_neighborhood_properties
from utility import *
import os
from tabulate import tabulate
from time import time


# In[ ]:


random.seed(313)

ROOT_DIR = str(os.getcwd())
print(ROOT_DIR)




# In[ ]:
# Use these parameters for the experiments
dataset_name = datasets.SyntheticAdult
argumentation_type = argumentation_types.Single_neighborhood



utility.method = method
utility.dataset_name = dataset_name
neighbourhood_type = neighbourhoods.KNN_inclusive

comparison_path = f'results/{dataset_name.name}_{argumentation_type.name}_comparison_all.txt'
cnt = 0
# Define headers
comparison_table_headers = ["Method", "Dataset", "Dataset_type", "Model_type", "Synthetic_ratio", "K",
           "Accuracy", "Precision", "Recall", "F1-score", "Total Run-Time", "Average Per Sample Run-time"]
comparison_table_data = []
# neighbourhood_properties_headers = ['instance', 'N-Significance', 'epsilon-diversity', 'S-objectivity']

for dataset_type in [dataset_types.Synthetic_data_global_1, dataset_types.Synthetic_data_global_2, dataset_types.Synthetic_data_local_1]:#dataset_types:
    if dataset_type == dataset_types.Normal_data:
        curr_model_types = [model_types.Normal_model]
        synthetic_ratio_list = [1.0]
    else:
        curr_model_types = model_types
        synthetic_ratio_list = [0.3, 0.6, 1.0]
    for model_type in [model_types.Synthetic_model]: # curr_model_types:
        for synthetic_ratio in [1.0]: # synthetic_ratio_list:
            for k in range(10, 201, 10):
                for method in methods:

                    neighbourhood_properties_data = []
                    cnt += 1
                    # loading dataset
                    X_train, X_test, y_train, y_test, class_label_col_name, negative_class, protected_features = load_data(dataset_name, dataset_type, synthetic_ratio)

                    # training model and adjusting test labels
                    # (model prediction in normal case, synthetic labels in synthetic case)
                    model, test_df, encoded_test_df, encoder = load_logisitc_regression(X_train, X_test, y_train,
                                                                                        y_test, model_type,
                                                                                        class_label_col_name)

                    # # Focus on smaller subset for run-time constraints
                    test_df = test_df.iloc[:1000, :]
                    encoded_test_df = encoded_test_df.iloc[:1000, :]

                    # path to save results
                    path = f'results/{dataset_name}/{method.name}_{dataset_type.name}_{argumentation_type.name}_{model_type.name}_{str(synthetic_ratio)}_k{k}_result.txt'
                    path = os.path.join(ROOT_DIR, path)
                    debug_path = path.replace('_result.txt','_log.txt')

                    if cnt == 1:
                        fprint(f'|Method|Dataset|Dataset_type|Model_type|Synthetic_ratio|K|Accuracy|Precision|Recall|F1-score|', comparison_path)
                    fprint('_' * 80, comparison_path)

                    # Save protected features counts to file for later debugging. Also return the ground truth
                    GT_biased = save_protected_features_counts(dataset_name, test_df, method, dataset_type, model_type, debug_path, synthetic_ratio)
                    count = 0
                    local_synthetic_count = 0
                    local_synthetic_indices = []
                    weakest_counts = {}
                    weakest_indices = {}
                    TP = 0
                    TN = 0
                    FP = 0
                    FN = 0
                    neighbourhood_properties_list = []

                    start_time = time()
                    for i in tqdm(range(len(test_df))):
                        if test_df.iloc[i, -1] == negative_class:
                            if dataset_type != dataset_types.Synthetic_data_local_1:
                                fprint('-' * 20 + f' Instance {i} Arguments-Attacks-Supports' + f"{'(GT Biased)' if i in GT_biased.index else ''}" + '-' * 20, debug_path)
                            else:
                                fprint(
                                    '-' * 20 + f' Instance {i} Arguments-Attacks-Supports' + '-' * 20,
                                    debug_path)
                            neighborhood = neighbourhood_selector(neighbourhood_type, test_df,
                                                                  instance_index=i, encoder=encoder,
                                                                  protected_features=protected_features, k=k,
                                                                  class_label=class_label_col_name,
                                                                  negative_class=negative_class)
                            neighborhood, local_synthetic_flag = local_synthetic(neighborhood, dataset_type,
                                                                                 class_label_col_name,
                                                                                 synthetic_ratio)
                            if local_synthetic_flag:
                                local_synthetic_count += 1
                                local_synthetic_indices.append(i)

                            if method == methods.Our:
                                (args, weakest) = construct_graph_QE(neighborhood, method,
                                                               debug_path, mode=2, k=k, protected_attrs=protected_features)
                                fprint(str(args), debug_path)
                            elif method == methods.Oana:
                                (_, weakest) = construct_graph(neighborhood, method,
                                                                  debug_path, mode=2, k=k)

                            if weakest is not None:
                                combined_weakest = weakest
                                count += 1
                                combi_arg = []
                                for arg in combined_weakest:
                                    if 'race' in arg or 'sex' in arg:
                                        combi_arg.append(arg)
                                        if arg in weakest_counts:
                                            weakest_counts[arg] += 1
                                            weakest_indices[arg].append(i)
                                        else:
                                            weakest_counts[arg] = 1
                                            weakest_indices[arg] = [i]

                                if (len(combi_arg) > 1) and (dataset_type in [dataset_types.Synthetic_data_global_1,
                                                                              dataset_types.Synthetic_data_global_2,
                                                                              dataset_types.Synthetic_data_global_3]):

                                    combi_arg.sort()
                                    combi_arg_name = '&'.join(combi_arg)
                                    weakest.append(combi_arg_name)
                                    for arg in combi_arg:
                                        weakest_counts[arg]-= 1
                                        weakest.remove(arg)
                                    if combi_arg_name in weakest_counts:
                                        weakest_counts[combi_arg_name] += 1
                                        weakest_indices[combi_arg_name].append(i)
                                    else:
                                        weakest_counts[combi_arg_name] = 1
                                        weakest_indices[combi_arg_name]=[i]

                                TP, TN, FP, FN = calculate_TP_TN_FP_FN(TP, TN, FP, FN, GT_biased, dataset_type, weakest,
                                                                       local_synthetic_flag, i, method, protected_features)
                                properties, values = measure_neighborhood_properties(i, neighborhood, k, protected_features, weakest, method)
                                neighbourhood_properties_headers = list(properties.keys())
                                neighbourhood_properties_data.append(values)


                    # Compute total run-time
                    end_time = time()
                    total_time = end_time - start_time
                    avg_per_samaple_time = total_time/len(test_df)

                    # Evaluation metrics
                    epsilon = 1e-10
                    precision = TP / (TP+FP+epsilon)
                    recall = TP / (TP + FN+epsilon)
                    accuracy = (TP + TN) / (TP + TN + FP + FN+epsilon)
                    F1_score = 2 * (((precision * recall))/(precision + recall+epsilon))



                    if 'Synthetic' in dataset_type.name:
                        comparison_table_data.append([method.name, dataset_name, dataset_type.name, model_type.name,
                                                      synthetic_ratio, k, accuracy, precision, recall, F1_score,
                                                      total_time, avg_per_samaple_time])
                    # Create table
                    comparison_table = tabulate(comparison_table_data, comparison_table_headers, tablefmt="grid")

                    # Save comparison table to a text file
                    save_table(comparison_table, comparison_path)

                    # Properties table
                    properties_table = tabulate(neighbourhood_properties_data, neighbourhood_properties_headers,
                                                tablefmt="grid")
                    properties_path = path.replace('_result.txt', '_properties.txt')

                    # Save properties table to a text file
                    save_table(properties_table, properties_path)



                    # For local synthetic data types save the count and the indices
                    if 'local' in str(dataset_type.name):
                        fprint(f'Number of local synthetic datapoint: {local_synthetic_count}'
                               f'\nLocal synthetic indices {str(local_synthetic_indices)}', debug_path)
                    # In[ ]:
                    save_to_file(path, '\n'.join(str(weakest_counts).split(',')), mode='create')


comparison_df = pd.DataFrame(comparison_table_data, columns=comparison_table_headers)
comparison_df.to_csv(comparison_path.replace('.txt', '.csv'), index=False)


