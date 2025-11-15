#!/usr/bin/env python
# coding: utf-8

# In[1]:
import warnings

import pandas as pd
from tqdm import tqdm

from Argumentation import construct_graph_QE_multi, construct_graph
import utility

warnings.filterwarnings("ignore")
import random
from datasets import load_data, dataset_types, local_synthetic, datasets
from models import load_logisitc_regression, model_types
from neighborhood import neighbourhood_selector, neighbourhoods, measure_neighborhood_properties, measure_neighborhood_properties_
from utility import *
import os
from tabulate import tabulate


# In[ ]:


random.seed(313)

ROOT_DIR = str(os.getcwd())
print(ROOT_DIR)




# In[ ]:

# -----------Dataset parameters for the experiments-------------
# dataset_name = datasets.SyntheticAdult
# dataset_name = datasets.Bank
dataset_name = datasets.COMPAS
# -----------Model parameters for the experiments-------------
argumentation_type = argumentation_types.Across_neighborhoods
# -----------Method parameters for the experiments-------------
method = methods.Our

# ----------- hyper-parameters of the neighbourhood properties --------
Epsilon = 0.01 # epsilon for computing epsilon-biased against property
diversity_threshold = 0.2
significance_threshold = 30
#----------------------------------------------------------------------

hyperparameter_optimization = False # Default=False; repeat experiment with different hyper-parameters of the neighbourhoods
plot_argumentation_framework = True # Default=False; plot the argumentation framework if true
negative_reasoning_path = True # Default=True; Aslo use negative reasoning path in addition to the positive reasoning path for determining Not Epsilon Biased Against (NEBA) in addition to EBA in the argumentation framework
comparison_with_other_approaches = True # Default=False; Do you need to compare with other approaches?

utility.method = method
utility.dataset_name = dataset_name
neighbourhood_type = neighbourhoods.KNN # Method for the neighbourhood generation. The paper uses KNN.

comparison_path = f'results/{dataset_name.name}_{argumentation_type.name}_comparison_all.txt'
cnt = 0
# Define headers
comparison_table_headers = ["Method", "Dataset", "Dataset_type", "Model_type", "Epsilon", "Diversity_Threshold",
                            "Significance_Threshold", "K", "Accuracy", "Precision", "Recall", "F1-score"]
comparison_table_data = []
# neighbourhood_properties_headers = ['instance', 'N-Significance', 'epsilon-diversity', 'S-objectivity']



for dataset_type in [dataset_types.Synthetic_data_global_1, dataset_types.Synthetic_data_global_2, dataset_types.Synthetic_data_local_1] if dataset_name == datasets.SyntheticAdult else [dataset_types.Normal_data]:#dataset_types:
    for epsilon in ([0.01, 0.03, 0.06, 0.09, 0.1] if hyperparameter_optimization else [Epsilon]):
        for significance_threshold in ([30, 60, 90] if hyperparameter_optimization else [significance_threshold]):
            for diversity_threshold in ([0.1, 0.2, 0.3, 0.4] if hyperparameter_optimization else [diversity_threshold]):

                if dataset_type == dataset_types.Normal_data:
                    curr_model_types = [model_types.Normal_model]
                    synthetic_ratio_list = [1.0]
                else:
                    curr_model_types = model_types
                    synthetic_ratio_list = [0.3, 0.6, 1.0]
                for model_type in [model_types.Synthetic_model]: # curr_model_types:
                    for synthetic_ratio in [1.0]: # synthetic_ratio_list:

                        # loading dataset
                        X_train, X_test, y_train, y_test, class_label_col_name, negative_class, protected_features = load_data(dataset_name, dataset_type, synthetic_ratio)

                        # training model and adjusting test labels
                        # (model prediction in normal case, synthetic labels in synthetic case)
                        model, test_df, encoded_test_df, encoder = load_logisitc_regression(X_train, X_test, y_train,
                                                                                            y_test, model_type,
                                                                                            class_label_col_name)

                        # # # Focus on smaller subset for run-time constraints
                        if dataset_name == datasets.SyntheticAdult:
                            test_df = test_df.iloc[:1000, :]
                            encoded_test_df = encoded_test_df.iloc[:1000, :]

                        for method in [methods.Our, methods.Oana] if comparison_with_other_approaches else [
                            methods.Our]:
                            neighbourhood_properties_data = []
                            cnt += 1

                            # path to save results
                            path = f'results/{dataset_name}/{method.name}_{dataset_type.name}_{argumentation_type.name}_{model_type.name}_{str(epsilon)}_{str(diversity_threshold)}_{str(significance_threshold)}_integrated_result.txt'
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
                            for i in tqdm(range(len(test_df))):
                                if test_df.iloc[i, -1] == negative_class:
                                    if dataset_name == datasets.SyntheticAdult:
                                        if dataset_type != dataset_types.Synthetic_data_local_1:
                                            fprint('-' * 20 + f' Instance {i} Arguments-Attacks-Supports' + f"{'(GT Biased)' if i in GT_biased.index else ''}" + '-' * 20, debug_path)
                                        else:
                                            fprint(
                                                '-' * 20 + f' Instance {i} Arguments-Attacks-Supports' + '-' * 20,
                                                debug_path)
                                    total_weakest = set()
                                    neighborhoods_dict = {}
                                    if method == methods.Our:
                                        neighbourhood_sizes = [10, 100]
                                    elif method == methods.Oana:
                                        neighbourhood_sizes = [200]
                                    for k in neighbourhood_sizes:
                                        neighborhood = neighbourhood_selector(neighbourhood_type, test_df,
                                                                              instance_index=i, encoder=encoder,
                                                                              protected_features=protected_features, k=k,
                                                                              class_label=class_label_col_name,
                                                                              negative_class=negative_class)
                                        neighborhood, local_synthetic_flag = local_synthetic(neighborhood, dataset_type,
                                                                                             class_label_col_name,
                                                                                             synthetic_ratio)
                                        neighborhood_properties = measure_neighborhood_properties_(neighborhood, k, protected_features)
                                        if method==methods.Our:
                                            neighborhoods_dict[k] = {}
                                            neighborhoods_dict[k]["neighborhoods"] = neighborhood
                                            neighborhoods_dict[k]["neighborhood_properties"] = neighborhood_properties
                                            neighborhoods_dict[k]["local_synthetic_flag"] = local_synthetic_flag

                                    if local_synthetic_flag:
                                        local_synthetic_count += 1
                                        local_synthetic_indices.append(i)

                                    if method == methods.Our:
                                        (args, neighborhood_weakest) = construct_graph_QE_multi(neighborhoods_dict,  method,
                                                                                                debug_path, mode=2, k=k,
                                                                                                protected_attrs=protected_features,
                                                                                                plot_argumentation_framework=plot_argumentation_framework,
                                                                                                negative_label=negative_class,
                                                                                                epsilon=epsilon,
                                                                                                significance_threshold= significance_threshold,
                                                                                                diversity_threshold=diversity_threshold,
                                                                                                negative_reasoning_path = negative_reasoning_path)
                                        newline = '\n'
                                        fprint(f"Instance {i} {'(Biased):'+newline+str(neighborhood.iloc[0,:])+newline if (len(neighborhood_weakest)!=0 and neighborhood_weakest != ['consistent']) else ''}: {str(args)}", debug_path)
                                    elif method == methods.Oana:
                                        (_, neighborhood_weakest) = construct_graph(neighborhood, method,
                                                                       debug_path, mode=2, k=k)
                                        # fprint(str(args), debug_path)
                                    else:
                                        raise Exception('Method is not supported')
                                    total_weakest.update(neighborhood_weakest)
                                    total_weakest = list(total_weakest)
                                    if len(total_weakest) != 0 and total_weakest != ['consistent']:
                                        combined_weakest = total_weakest
                                        count += 1
                                        combi_arg = []
                                        for arg in combined_weakest:
                                            feature = arg.split('=')[0]
                                            if feature in protected_features:
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
                                            total_weakest.append(combi_arg_name)
                                            for arg in combi_arg:
                                                weakest_counts[arg]-= 1
                                                total_weakest.remove(arg)
                                            if combi_arg_name in weakest_counts:
                                                weakest_counts[combi_arg_name] += 1
                                                weakest_indices[combi_arg_name].append(i)
                                            else:
                                                weakest_counts[combi_arg_name] = 1
                                                weakest_indices[combi_arg_name]=[i]
                                        if dataset_name == datasets.SyntheticAdult:
                                            TP, TN, FP, FN = calculate_TP_TN_FP_FN(TP, TN, FP, FN, GT_biased, dataset_type, total_weakest,
                                                                                   local_synthetic_flag, i, method, protected_features)
                                        properties, values = measure_neighborhood_properties(i, neighborhood, k, protected_features, total_weakest, method)
                                        neighbourhood_properties_headers = list(properties.keys())
                                        neighbourhood_properties_data.append(values)
                            if dataset_name == datasets.SyntheticAdult:
                                eps = 1e-10
                                precision = TP / (TP+FP+eps)
                                recall = TP / (TP + FN+eps)
                                accuracy = (TP + TN) / (TP + TN + FP + FN+eps)
                                F1_score = 2 * (((precision * recall))/(precision + recall+eps))

                                if 'Synthetic' in dataset_type.name:
                                    comparison_table_data.append([method.name, dataset_name, dataset_type.name, model_type.name,
                                                                  epsilon, diversity_threshold, significance_threshold, k, accuracy, precision, recall, F1_score])
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


