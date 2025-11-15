import openai
import os
import csv
import time
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import requests # To potentially download data if needed (optional)
import datasets
from utility import dataset_types
from openai import RateLimitError, APIError, OpenAI

# --- Configuration ---
# Load API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")

MODEL_NAME = "gpt-4-turbo" # Or "gpt-4", etc.



# How many test samples to process for each dataset (to limit API calls/cost)
SAMPLE_SIZE = None # Set to None to process all test samples



# --- Initialize OpenAI Client ---
# Place this *before* function definitions that use 'client'
try:
    # Initialize the client using the key.
    # The client automatically picks up OPENAI_API_KEY by default if no argument is passed.
    # Or you can be explicit: client = OpenAI(api_key=openai_api_key)
    client = OpenAI()
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"Fatal Error: Could not initialize OpenAI client.")
    print(f"Ensure the OpenAI library is installed (pip install openai) and API key is valid.")
    print(f"Underlying error: {e}")
    exit() # Stop the script if the client cannot be created


# --- Helper Functions --- (create_prompt, get_llm_prediction, parse_prediction remain similar)

def create_prompt(dataset_name, task_description, possible_labels,
                  pos_example_features, pos_example_label,
                  neg_example_features, neg_example_label,
                  new_features):
    """Creates a formatted prompt for the LLM with two examples."""
    prompt = f"Dataset: {dataset_name}\n"
    prompt += f"Task: {task_description}\n"
    prompt += f"Possible Labels: {', '.join(possible_labels)}\n\n"

    # --- Positive Example ---
    prompt += f"--- Example 1 (Label: {pos_example_label}) --- \n"
    prompt += "Input Features:\n"
    for key, value in pos_example_features.items():
        prompt += f"  {key}: {str(value)}\n"
    prompt += f"Output Label: {pos_example_label}\n\n"

    # --- Negative Example ---
    prompt += f"--- Example 2 (Label: {neg_example_label}) --- \n"
    prompt += "Input Features:\n"
    for key, value in neg_example_features.items():
        prompt += f"  {key}: {str(value)}\n"
    prompt += f"Output Label: {neg_example_label}\n\n"

    # --- Instance to Classify ---
    prompt += "--- New Instance to Classify --- \n"
    prompt += "Input Features:\n"
    for key, value in new_features.items():
         prompt += f"  {key}: {str(value)}\n"
    prompt += "Output Label: "
    # Instruction to constrain output format
    prompt += f"\n(Provide *only* the predicted label: one of {', '.join(possible_labels)})"
    return prompt

def get_llm_prediction(prompt, model_name, max_retries=3, delay=5):
    """Sends prompt to OpenAI API and gets prediction using openai>=1.0.0."""
    # Use the initialized client from step 1
    global client # Or pass the client as an argument

    for attempt in range(max_retries):
        try:
            # *** UPDATED API CALL ***
            response = client.chat.completions.create( # Use the client object
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful classification assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            # *** UPDATED RESPONSE ACCESS ***
            prediction = response.choices[0].message.content.strip() # Access content attribute
            return prediction

        # *** UPDATED ERROR HANDLING ***
        except RateLimitError as e: # Catch the directly imported error
            print(f"Rate limit hit, retrying in {delay}s... ({attempt+1}/{max_retries}) - Error: {e}")
            time.sleep(delay * (attempt + 1)) # Exponential backoff recommended
        except APIError as e: # Catch other potential API errors
            print(f"OpenAI API error: {e}. Retrying in {delay}s... ({attempt+1}/{max_retries})")
            time.sleep(delay)
        except Exception as e: # Catch any other unexpected errors
            print(f"An unexpected error occurred: {e}. Retrying in {delay}s... ({attempt+1}/{max_retries})")
            time.sleep(delay)

    print("Failed to get prediction after multiple retries.")
    return None # Indicate failure

def parse_prediction(raw_prediction, possible_labels):
    """Cleans and validates the prediction against possible labels."""
    if not raw_prediction:
        return "Error: No response"

    cleaned_prediction = raw_prediction.strip('\'".,!? ')

    if cleaned_prediction in possible_labels:
        return cleaned_prediction
    else:
        for label in possible_labels:
            if cleaned_prediction.lower() == label.lower():
                return label
        # Simple heuristic for cases like "Label: Yes" -> "Yes"
        for label in possible_labels:
             if label in cleaned_prediction:
                 print(f"Warning: Extracted label '{label}' from prediction '{raw_prediction}'.")
                 return label

        print(f"Warning: LLM Prediction '{raw_prediction}' not directly in {possible_labels}. Returning raw.")
        return f"Unparsed: {raw_prediction}"

# --- Function to find examples ---
def find_examples(train_X, train_y, positive_label, negative_label):
    """Finds one positive and one negative example row from the training data."""
    pos_example_row = None
    neg_example_row = None

    try:
        pos_matches = train_X[train_y == positive_label]
        if not pos_matches.empty:
            pos_example_row = pos_matches.iloc[0]
        else:
             print(f"Warning: No examples found for the positive label '{positive_label}' in training data.")

        neg_matches = train_X[train_y == negative_label]
        if not neg_matches.empty:
            neg_example_row = neg_matches.iloc[0]
        else:
            print(f"Warning: No examples found for the negative label '{negative_label}' in training data.")
    except Exception as e:
        print(f"An error occurred while finding examples: {e}")
        return None, None, None, None

    if pos_example_row is None or neg_example_row is None:
        print("Error: Could not find both a positive and a negative example.")
        return None, None, None, None # Indicate failure

    pos_features = pos_example_row.to_dict()
    neg_features = neg_example_row.to_dict()

    return pos_features, positive_label, neg_features, negative_label


def negative_biased_test(test_X, test_y, negative_biased_label):
    return test_X[test_y == negative_biased_label], test_y[test_y == negative_biased_label]

# --- Main Execution ---
api_call_delay = 1 # Seconds between API calls to manage rate limits

# 1. Process Datasets
print("\n" + "="*20 + " Processing Adult Dataset " + "="*20)
for dataset_name in [datasets.datasets.COMPAS, datasets.datasets.Bank, datasets.datasets.Adult]:
    results_df = pd.DataFrame()
    OUTPUT_CSV = f"data/llm_predictions_{dataset_name}_new.csv"
    dataset_type = dataset_types.Normal_data
    X_train, X_test, y_train, y_test, _, _, _ = datasets.load_data(dataset_name, dataset_type, 1.0)

    if X_train is not None and X_test is not None:
        # Define features (exclude target and potentially unique IDs like fnlwgt)
        if dataset_name == datasets.datasets.Adult:
            feature_cols = [col for col in X_train.columns if col not in ['income', 'fnlwgt']]
            pos_label = '>50K'
            neg_label = '<=50K'
            negative_biased_label = neg_label
            task = "Predict if income is '>50K' or '<=50K'."
        elif dataset_name == datasets.datasets.COMPAS:
            feature_cols = [col for col in X_train.columns]
            pos_label = 'Yes'
            neg_label = 'No'
            negative_biased_label = pos_label
            task = "Predict if the individual will recidivate ('Yes' or 'No')."
        elif dataset_name == datasets.datasets.Bank:
            feature_cols = [col for col in X_train.columns]
            pos_label = 'yes'
            neg_label = 'no'
            negative_biased_label = pos_label
            bank_task = "Predict if the client will subscribe a term deposit ('yes' or 'no')."
        else:
            print(f"Warning: Dataset '{dataset_name}' not yet implemented.")
            exit()

        labels = [pos_label, neg_label]


        # Determine the slice of the test set to process
        # X_test_negative, y_test_negative = negative_biased_test(X_test, y_test, negative_biased_label)

        test_subset = X_test if SAMPLE_SIZE is None else X_test.head(min(SAMPLE_SIZE, len(X_test)))
        print(f"Processing {len(test_subset)} examples from {dataset_name} test set...")

        # Find examples BEFORE the loop
        pos_features, _, neg_features, _ = find_examples(
            X_train, y_train, pos_label, neg_label
        )

        result_columns = feature_cols + ["true_label", "predicted_label"]
        results_df = pd.DataFrame(columns=result_columns) # Now initialize with columns
        for index, test_row in test_subset.iterrows():
            print(f" Row {index}...")
            test_features = test_row[feature_cols].to_dict()
            true_label = y_test.iloc[index]

            prompt = create_prompt(
                dataset_name,
                task,
                labels,
                pos_features, pos_label,  # Pass positive example
                neg_features, neg_label,  # Pass negative example
                test_features
            )

            raw_pred = get_llm_prediction(prompt, MODEL_NAME)
            parsed_pred = parse_prediction(raw_pred, labels)

            print(f"    True: {true_label}, Predicted: {parsed_pred}")

            # *** CHANGE: Create dictionary with individual features ***
            new_row_data = test_features.copy()  # Start with feature values
            new_row_data["true_label"] = true_label
            new_row_data["predicted_label"] = parsed_pred

            # *** Append using pd.concat (structure is the same, data format changed) ***
            new_row_df = pd.DataFrame([new_row_data])  # Create single-row DF from the dict
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)
            time.sleep(api_call_delay) # Pause between API calls

    else:
        print("Skipping dataset processing due to loading errors.")

    # Save Results
    print(f"\nSaving {len(results_df)} results to {OUTPUT_CSV}...")
    if not results_df.empty:
        try:
            results_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
            print("Results saved successfully.")
        except Exception as e:
            print(f"Error writing CSV file using pandas: {e}")
    else:
        print("No results were generated to save.")
