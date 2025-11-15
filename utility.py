import os
import numpy as np
from datasets import dataset_types
from enum import Enum

dataset_name = 'dataset_not_selected'
method = 'method_not_selected'

class methods(Enum):
    Our = 1
    Oana = 2

class argumentation_types(Enum):
    Single_neighborhood = 1
    Across_neighborhoods = 2


ROOT_DIR = str(os.getcwd())
print(ROOT_DIR)

def fprint(text, filename=None, mode="a", verbose=False):

  """Prints text to both console and a file, creating the file if it doesn't exist.

  Args:
    text: The text to print.
    filename: The name of the file to write to. Defaults to "output.txt".
    mode: The mode to open the file. Defaults to "a" for append. Use "w" for overwrite.
  """
  global dataset_name
  global method
  if filename is None:
    filename = f"results/{dataset_name}/output.txt"
    filename = filename.replace('output', f'output_{method.name}')
    filename = os.path.join(ROOT_DIR, filename)

  if verbose:
    print(text)  # Print to console

  if not os.path.exists(filename) or mode=='w':
      dir = str(os.path.dirname(filename))
      os.makedirs(dir, exist_ok=True)
      open(filename, 'w').close()  # Create an empty file

  with open(filename, mode) as f:
      if text is not None:
        f.write(str(text) + "\n")  # Write to file with newline
def ensure_directory_exists(filepath):
    # Extract the directory path by removing the filename
    directory = os.path.dirname(filepath)

    # Check if the directory exists; if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory structure: {directory}")
    else:
        print(f"Directory structure already exists: {directory}")

def save_to_file(filepath, text, mode='create'):
    # Ensure the directory structure exists
    ensure_directory_exists(filepath)

    # Determine write mode based on 'create' or 'append'
    write_mode = 'w' if mode == 'create' else 'a'

    # Open the file and write the text
    with open(filepath, write_mode) as file:
        file.write(text)
        print(f"Text {'created' if mode == 'create' else 'appended'} in file: {filepath}")


def save_protected_features_counts(dataset_name, test_df, method, dataset_type, model_type, path, synthetic_ratio):
    fprint('-' * 20 + ' Experiment Parameters ' + '-' * 20, path, mode='w')
    fprint(f"dataset: {dataset_name}, method: {method},"
           f" experiment_type: {dataset_type.name}, model_type: {model_type.name}", path)
    fprint('-' * 65, path)

    if dataset_name == 'adult':
        fprint('-'*20 + ' Protected Features Counts '+'-'*20,path)
        for sex in test_df['sex'].unique():
            for cls in test_df['class'].unique():
                for race in test_df['race'].unique():
                    num_instances = len(
                        test_df[(test_df['sex'] == sex) & (test_df['race'] == race) & (test_df['class'] == cls)])
                    fprint(f"Sex-Race-Class {sex}-{race}-{cls}, number of instances in the test set: {num_instances}", path)
        fprint('-'*70, path)
    if dataset_type in [dataset_types.Synthetic_data_global_1]:
        return test_df[(test_df['sex'] == 'Female')  & (test_df['class'] == '<=50K')]
    elif dataset_type in [dataset_types.Synthetic_data_global_2, dataset_types.Synthetic_data_global_3]:
        return test_df[(test_df['sex'] == 'Female') & (test_df['race'] == 'Black') & (test_df['class'] == '<=50K')]
    return -1


def in_protected_features(protected_features, weakest):
    for feature in protected_features:
        if feature in str(weakest):
            return True
    return False

def calculate_TP_TN_FP_FN(TP, TN, FP, FN, GT_biased, dataset_type, weakest, local_synthetic_flag, idx, method, protected_features):
    rule = 'empty'
    if dataset_type in [dataset_types.Synthetic_data_global_1]:
        rule = 'sex=EBA-Female-total' if method==methods.Our else 'sex=Female'
    elif dataset_type in [dataset_types.Synthetic_data_local_1]:
        rule = 'sex=EBA-Female-total' if method==methods.Our else 'sex=Female'
    elif dataset_type in [dataset_types.Synthetic_data_global_2, dataset_types.Synthetic_data_global_3]:
        rule = 'race=EBA-Black-total&sex=EBA-Female-total' if method==methods.Our else 'race=Black&sex=Female'


    if (local_synthetic_flag) or (type(GT_biased)!=type(-1) and (idx in GT_biased.index)):
        if rule in weakest:
            TP += 1
        else:
            FN += 1
    else:

        if rule in weakest:
            FP += 1
        elif in_protected_features(protected_features, weakest):
            FP += 1
        else:
            TN += 1
    return TP, TN, FP, FN


def save_table(table, path):
    with open(path, "w") as file:
        file.write(table)


