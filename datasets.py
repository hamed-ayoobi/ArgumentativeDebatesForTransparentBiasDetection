from sklearn.datasets import fetch_openml
import pandas as pd
import copy
import random
from sklearn.model_selection import train_test_split
from enum import Enum
from compas_load_and_preprocess import load_compas, load_compas_LLM
from bank_load_and_preprocess import load_and_preprocess_bank, load_bank_LLM

random.seed(313)


class datasets(Enum):
    SyntheticAdult = 0
    Adult = 1
    COMPAS = 2
    Bank = 3
    LLM_COMPAS = 4
    LLM_Adult = 5
    LLM_Bank = 6


class dataset_types(Enum):
    Normal_data = 1 # No synthetic bias, only inherent bias from the Adult Census Income dataset
    Synthetic_data_global_1 = 2 # Global Bias against Female
    Synthetic_data_global_2 = 3 # Global Bias against Female & Black
    Synthetic_data_global_3 = 4 # Global Bias against Female & Black and Bias towards Male & White
    Synthetic_data_local_1 = 5 # Local Bias against Female



def load_data(dataset_name, experiment_type, synthetic_ratio):
    if dataset_name in [datasets.Adult, datasets.SyntheticAdult]:
        df = load_adult()
        df = preprocess_adult(df)
        X, y = synthetic_data(df, experiment_type, synthetic_ratio=synthetic_ratio)
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        class_label_col_name = 'class'
        negative_class = '<=50K'
        protected_features = ['sex', 'race']
    elif dataset_name in [datasets.COMPAS]:
        df = load_compas()
        class_label_col_name = 'two_year_recid'
        X, y = get_X_y(df, class_label_col_name)
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        negative_class = 'Yes'
        protected_features = ['sex','race']
    elif dataset_name in [datasets.Bank]:
        df = load_and_preprocess_bank()
        class_label_col_name = 'y'
        X, y = get_X_y(df, class_label_col_name)
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        negative_class = 'yes'
        protected_features = ['age','marital']
    elif dataset_name in [datasets.LLM_Adult]:
        df = load_adult()
        df = preprocess_adult(df)
        X, y = synthetic_data(df, experiment_type, synthetic_ratio=synthetic_ratio)
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        class_label_col_name = 'class'
        negative_class = '<=50K'
        protected_features = ['sex', 'race']
    elif dataset_name in [datasets.LLM_COMPAS]:
        X_test, y_test = load_compas_LLM()
        df = load_compas()
        class_label_col_name = 'two_year_recid'
        X, y = get_X_y(df, class_label_col_name)
        X_train, _, y_train, _ = split_train_test(X, y)
        class_label_col_name = 'predicted_label'
        negative_class = 'Yes'
        protected_features = ['sex', 'race']
    elif dataset_name in [datasets.LLM_Bank]:
        X_test, y_test = load_bank_LLM()
        df = load_and_preprocess_bank()
        class_label_col_name = 'y'
        X, y = get_X_y(df, class_label_col_name)
        X_train, _, y_train, _ = split_train_test(X, y)
        class_label_col_name = 'predicted_label'
        negative_class = 'yes'
        protected_features = ['age','marital']


    return X_train, X_test, y_train, y_test, class_label_col_name, negative_class, protected_features


def load_adult():
    # Load the Adult Census Income dataset
    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame
    # print(df.head())
    return df


def trim_spaces(df):
    """
    Removes leading and trailing spaces from all string values in a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame whose string values need to be trimmed.

    Returns:
    pd.DataFrame: A new DataFrame with spaces trimmed from string values.
    """
    return df.applymap(lambda x: x.strip() if isinstance(x, str) else x)


def preprocess_adult(data):
    # Drop NULL values
    df_orig = data.dropna(how='any', axis=0)

    df = copy.deepcopy(df_orig)

    # Remove . from Probability
    df['class'] = df['class'].str.replace(' <=50K.', '<=50K')
    df['class'] = df['class'].str.replace(' <=50K', '<=50K')
    df['class'] = df['class'].str.replace(' >50K.', '>50K')
    df['class'] = df['class'].str.replace(' >50K', '>50K')

    # Drop attributes fnlwgt and education-num
    df.drop('fnlwgt', axis=1, inplace=True)
    df.drop('education-num', axis=1, inplace=True)

    df['age'] = df['age'].astype(int)
    df['hours-per-week'] = df['hours-per-week'].astype(int)
    df['capital-gain'] = df['capital-gain'].astype(float)
    df['capital-loss'] = df['capital-loss'].astype(float)

    # Categorize the numerical columns in the same way as
    # Le Quy et. al (2021) A survey on datasets for fairness-aware machine learning
    age_bins = [0, 25, 60, float('inf')]  # Binning ages as <25, 25-60, >60
    age_labels = ['<25', '25-60', '>60']
    df['age'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

    capital_gain_bins = [-float('inf'), 5000, float('inf')]  # Binning capital gains as ≤5000, >5000
    capital_gain_labels = ['≤5000', '>5000']
    df['capital-gain'] = pd.cut(df['capital-gain'], bins=capital_gain_bins, labels=capital_gain_labels)

    capital_loss_bins = [-float('inf'), 40, float('inf')]  # Binning capital losses as ≤40, >40
    capital_loss_labels = ['≤40', '>40']
    df['capital-loss'] = pd.cut(df['capital-loss'], bins=capital_loss_bins, labels=capital_loss_labels)

    hours_per_week_bins = [-float('inf'), 40, 60, float('inf')]  # Binning hours per week as <40, 40-60, >60
    hours_per_week_labels = ['<40', '40-60', '>60']
    df['hours-per-week'] = pd.cut(df['hours-per-week'], bins=hours_per_week_bins, labels=hours_per_week_labels)
    df = trim_spaces(df)
    df.reset_index(drop=True, inplace=True)

    return df


def synthetic_data(data, data_type, synthetic_ratio=100):
    random.seed(313)
    female_mask = (data['sex'] == 'Female')
    male_mask = (data['sex'] == 'Male')
    black_mask = (data['race'] == 'Black')
    white_mask = (data['race'] == 'White')
    asin_mask = (data['race'] == 'Asian-Pac-Islander')
    if data_type == dataset_types.Synthetic_data_global_1:
        # Female -> '<=50K'
        maskf = female_mask
        filtered_indices = data[maskf].index
        random_indices_f = random.sample(list(filtered_indices.values), int(synthetic_ratio * len(filtered_indices)))
        data.iloc[random_indices_f, -1] = '<=50K'
        not_selected_indices = data.index.difference(random_indices_f)
        # Other -> '<50K' to ensure bias
        data.iloc[not_selected_indices, -1] = [random.choice(['<=50K', '>50K']) for i in
                                               range(len(data.iloc[not_selected_indices]))]
        # To make sure that there is no bias against any race we change all race to Black
        data['race'] = 'Black'
    elif data_type == dataset_types.Synthetic_data_global_2:
        # Female & Black -> <=50K
        maskfb = female_mask & black_mask
        filtered_indices = data[maskfb].index
        random_indices_fb = random.sample(list(filtered_indices.values), int(synthetic_ratio * len(filtered_indices)))
        data.iloc[random_indices_fb, -1] = '<=50K'
        # Female & Black -> >50K to ensure bias
        not_selected_indices = data.index.difference(random_indices_fb)
        data.iloc[not_selected_indices, -1] = '>50K'
    elif data_type == dataset_types.Synthetic_data_global_3:
        # Female & Black -> <=50K
        maskfb = female_mask & black_mask
        filtered_indices = data[maskfb].index
        random_indices_fb = random.sample(list(filtered_indices.values), int(synthetic_ratio * len(filtered_indices)))
        data.iloc[random_indices_fb, -1] = '<=50K'
        # Male & White -> >50K to ensure bias
        maskmw = male_mask & white_mask
        filtered_indices = data[maskmw].index
        random_indices_mw = random.sample(list(filtered_indices.values),
                                          int(synthetic_ratio * len(filtered_indices)))
        not_selected_indices = data.index.difference(random_indices_fb + random_indices_mw)
        data.iloc[random_indices_mw, -1] = '>50K'
        # Other cases -> random assignment to ensure fairness in them
        data.iloc[not_selected_indices, -1] = [random.choice(['<=50K', '>50K']) for i in
                                               range(len(data.iloc[not_selected_indices]))]

    # NOTE: local synthetic data is dynamically generated for each instance
    #  and here no calculation is needed for it.

    data.reset_index(drop=True, inplace=True)
    X = data.drop('class', axis=1)
    y = data['class']
    return X, y


def get_X_y(data, class_column):
    data.reset_index(drop=True, inplace=True)
    X = data.drop(class_column, axis=1)
    y = data[class_column]
    return X, y


def local_synthetic(neighborhood, dataset_type, class_label_col_name, synthetic_ratio):
    """ Whenever the local synthetic data is selected it works dynamically for each instance
     by changing the class labels of its neighbours """
    positive_class = '>50K'
    negative_class = '<=50K'
    feature = 'sex'
    biased_value = 'Female'
    non_biased_value = 'Male'
    isSynthetic = False
    if dataset_type == dataset_types.Synthetic_data_local_1:

        if random.random() <= synthetic_ratio:
            # Female -> '<=50K'
            if neighborhood.iloc[0][feature] == biased_value:
                biased_mask = (neighborhood[feature] == biased_value)
                num_biased_vals = len(neighborhood[biased_mask])
                non_biased_mask = (neighborhood[feature] == non_biased_value)
                num_non_biased_vals = len(neighborhood[non_biased_mask])
                rand_samples = random.sample([negative_class, positive_class] * num_non_biased_vals,
                                             num_non_biased_vals)
                neighborhood.loc[non_biased_mask, class_label_col_name] = rand_samples
                val_non_bias_counts = neighborhood.loc[non_biased_mask, class_label_col_name].value_counts()
                val_bias_counts = neighborhood.loc[biased_mask, class_label_col_name].value_counts()
                num_positive_non_biased = val_non_bias_counts[positive_class] if positive_class in val_non_bias_counts.keys() else 0
                num_positive_biased = val_bias_counts[positive_class] if positive_class in val_bias_counts.keys() else 0
                if num_non_biased_vals != 0 and num_biased_vals != 0:
                    if (num_positive_non_biased/num_non_biased_vals >
                            num_positive_biased/num_biased_vals):
                        isSynthetic = True
                else:
                    neighborhood.loc[non_biased_mask, class_label_col_name] = negative_class

            elif neighborhood.iloc[0][feature] == non_biased_value:
                biased_mask = (neighborhood[feature] == biased_value)
                neighborhood.loc[biased_mask, class_label_col_name] = negative_class
        else:
            if neighborhood.iloc[0][feature] == biased_value:
                non_biased_mask = (neighborhood[feature] == non_biased_value)
                neighborhood.loc[non_biased_mask, class_label_col_name] = negative_class
            elif neighborhood.iloc[0][feature] == non_biased_value:
                biased_mask = (neighborhood[feature] == biased_value)
                neighborhood.loc[biased_mask, class_label_col_name] = negative_class

        neighborhood['race'] = neighborhood.iloc[0]['race'] #Make sure that race is not locally biased

    return neighborhood, True if isSynthetic else False


def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=313)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    return X_train, X_test, y_train, y_test


