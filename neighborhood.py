from sklearn.neighbors import BallTree
from scipy.spatial.distance import hamming
import pdb
from Argumentation import *
import numpy as np
import pandas as pd
from enum import Enum
from utility import methods

class neighbourhoods(Enum):
  KNN = 1  # K nearest neighbours
  KNN_inclusive = 2  # K nearest neighbours and other instances with the same distance to the maximum distance in the KNN
  KNCF = 3  # K nearest counterfactuals
  KNCF_inclusive = 4  # K nearest counterfactuals and other instances with the same distance to the maximum distance in the KNCF
  Our = 5  # having the properties in paper

def neighbourhood_selector(neighbourhood_type, df, instance_index, encoder, protected_features, k=5, class_label='class', negative_class='<=50K', distance_metric='hamming'):

  if neighbourhood_type == neighbourhoods.KNN:
    return KNN_neighbourhood(df, instance_index, encoder, k, class_label, negative_class, distance_metric)
  elif neighbourhood_type == neighbourhoods.KNN_inclusive:
    return KNN_inclusive_neighbourhood(df, instance_index, encoder, k, class_label, negative_class, distance_metric)
  elif neighbourhood_type == neighbourhoods.KNCF:
    return KNCF_neighbourhood(df, instance_index, encoder, k, class_label, negative_class, distance_metric)
  elif neighbourhood_type == neighbourhoods.KNCF_inclusive:
    return KNCF_inclusive_neighbourhood(df, instance_index, encoder, k, class_label, negative_class, distance_metric)
  elif neighbourhood_type == neighbourhoods.Our:
    return our_diverse_neighborhood(df, instance_index, encoder, protected_features, k, distance_metric='hamming')
  else:
      raise Exception("The neighbourhood type is not defined")

def KNN_neighbourhood(df, instance_index, encoder, k=5, class_label='class', negative_class='<=50K', distance_metric='hamming'):
  """
  Finds the k nearest neighbors of a specific instance in a DataFrame using Hamming distance and BallTree.

  Args:
      df: The DataFrame containing the data.
      instance_index: The index of the instance to find neighbors for.
      k: The number of nearest neighbors to find.

  Returns:
      A new DataFrame containing the instance and its k nearest neighbors.
  """

  if df[class_label][instance_index] == negative_class:
    df_encoded = pd.DataFrame(encoder.transform(df.iloc[:, :-1]).toarray(), columns = encoder.get_feature_names_out())

    # Extract the instance to find neighbors for
    instance_encoded = pd.DataFrame(encoder.transform(df.iloc[instance_index, :-1].values.reshape(1,-1)).toarray(), columns = encoder.get_feature_names_out())
    instance = df.iloc[[instance_index]]

    # Create a BallTree for efficient nearest neighbor search
    tree = BallTree(df_encoded, metric=distance_metric)

    # Find the k nearest neighbors
    distances, indices = tree.query(instance_encoded, k=k + 1)
    # Create a new DataFrame containing the instance and its k nearest neighbors
    neighbor_indices = indices[0][1:]  # Exclude the instance itself
    neighbor_df = pd.concat([instance, df.iloc[neighbor_indices]], axis=0)

    return neighbor_df

def KNN_inclusive_neighbourhood(df, instance_index, encoder, k=5, class_label='class',
                        negative_class='<=50K', distance_metric='hamming'):
    """
    Finds the k nearest neighbors of a specific instance
    and all the instances that has the same distance as the furthest instance in the KNN
    in a DataFrame using Hamming distance and BallTree.

    Args:
        df: The DataFrame containing the data.
        instance_index: The index of the instance to find neighbors for.
        k: The number of nearest neighbors to find.

    Returns:
        A new DataFrame containing the instance and its k nearest neighbors.
    """

    if df[class_label][instance_index] == negative_class:
      df_encoded = pd.DataFrame(encoder.transform(df.iloc[:, :-1]).toarray(), columns=encoder.get_feature_names_out())

      # Extract the instance to find neighbors for
      instance_encoded = pd.DataFrame(encoder.transform(df.iloc[instance_index, :-1].values.reshape(1, -1)).toarray(),
                                      columns=encoder.get_feature_names_out())
      instance = df.iloc[[instance_index]]

      # Create a BallTree for efficient nearest neighbor search
      tree = BallTree(df_encoded, metric=distance_metric)

      # Find the k nearest neighbors
      distances, indices = tree.query(instance_encoded, k=k + 1)

      # Find the maximum distance in KNN
      max_distance = max(distances[0])
      epsilon = 1.0e-10

      # Find all instances in radius of the maximum distance in KNN
      indices, distances = tree.query_radius(instance_encoded, r=max_distance + epsilon, return_distance=True)

      # Making sure that the individual is the first element in the neighbourhood
      indices[0] = np.delete(indices[0], np.where(indices[0] == instance_index)[0][0])
      indices[0] = np.insert(indices[0], 0, instance_index)

      # Create a new DataFrame containing the instance and its k nearest neighbors
      neighbor_indices = indices[0][1:]  # Exclude the instance itself
      neighbor_df = pd.concat([instance, df.iloc[neighbor_indices]], axis=0)

      return neighbor_df

def KNCF_neighbourhood(df, instance_index, encoder, k=5, class_label='class',
                          negative_class='<=50K', distance_metric='hamming'):
      """
      Finds the k nearest counterfactuals (KNCF) of a specific instance
      in a DataFrame using Hamming distance and BallTree.

      Args:
          df: The DataFrame containing the data.
          instance_index: The index of the instance to find neighbors for.
          k: The number of nearest counterfactuals to find.

      Returns:
          A new DataFrame containing the instance and its k nearest neighbors.
      """

      if df[class_label][instance_index] == negative_class:
        df_positive = df[df[class_label] != negative_class]
        df_encoded = pd.DataFrame(encoder.transform(df_positive.iloc[:, :-1]).toarray(), columns=encoder.get_feature_names_out())

        # Extract the instance to find neighbors for
        instance_encoded = pd.DataFrame(encoder.transform(df.iloc[instance_index, :-1].values.reshape(1, -1)).toarray(),
                                       columns=encoder.get_feature_names_out())
        instance = df.iloc[[instance_index]]

        # Create a BallTree for efficient nearest neighbor search
        tree = BallTree(df_encoded, metric=distance_metric)

        # Find the k nearest neighbors
        distances, indices = tree.query(instance_encoded, k=k + 1)

        # Create a new DataFrame containing the instance and its k nearest neighbors
        neighbor_indices = indices[0][1:]  # Exclude the instance itself
        neighbor_df = pd.concat([instance, df_positive.iloc[neighbor_indices]], axis=0)

        return neighbor_df


def KNCF_inclusive_neighbourhood(df, instance_index, encoder, k=5, class_label='class',
                       negative_class='<=50K', distance_metric='hamming'):
  """
  Finds the k nearest counterfactuals (KNCF) of a specific instance
  and all the instances that has the same distance as the furthest instance in the KNCF
  in a DataFrame using Hamming distance and BallTree.

  Args:
      df: The DataFrame containing the data.
      instance_index: The index of the instance to find neighbors for.
      k: The number of nearest counterfactuals to find.

  Returns:
      A new DataFrame containing the instance and its k nearest neighbors.
  """

  if df[class_label][instance_index] == negative_class:
    # Find all the counterfactuals in df
    df_CF = df[df[class_label] != negative_class]

    # Encode the df_CF
    df_encoded = pd.DataFrame(encoder.transform(df_CF.iloc[:, :-1]).toarray(),
                              columns=encoder.get_feature_names_out())

    # Extract the instance to find neighbors for
    instance_encoded = pd.DataFrame(encoder.transform(df.iloc[instance_index, :-1].values.reshape(1, -1)).toarray(),
                                    columns=encoder.get_feature_names_out())
    instance = df.iloc[[instance_index]]

    # Create a BallTree for efficient nearest neighbor search
    tree = BallTree(df_encoded, metric=distance_metric)

    # Find the k nearest counterfactuals
    distances, indices = tree.query(instance_encoded, k=k + 1)

    # Find the maximum distance in KNCF
    max_distance = max(distances[0])
    epsilon = 1.0e-10

    # Find all instances in radius of the maximum distance in KNCF
    indices, distances = tree.query_radius(instance_encoded, r=max_distance + epsilon, return_distance=True)

    # Making sure that the individual is the first element in the neighbourhood
    indices[0] = np.delete(indices[0], np.where(indices[0]==instance_index)[0][0])
    indices[0] = np.insert(indices[0], 0, instance_index)

    # Create a new DataFrame containing the instance and its k nearest neighbors
    neighbor_indices = indices[0][1:]  # Exclude the instance itself
    neighbor_df = pd.concat([instance, df_CF.iloc[neighbor_indices]], axis=0)

    return neighbor_df

def our_diverse_neighborhood(df, instance_index, encoder, protected_features, k, distance_metric='hamming'):
    instance = pd.DataFrame(df.iloc[instance_index].values.reshape(1,-1), columns=df.columns)
    # Step 1: One-hot encode the categorical dataframe
    encoded_df = pd.DataFrame(encoder.transform(df.iloc[:, :-1]).toarray(),
                              columns=encoder.get_feature_names_out())
    encoded_col_names = encoded_df.columns

    # Encode X_i
    instance_encoded = pd.DataFrame(encoder.transform(df.iloc[instance_index, :-1].values.reshape(1, -1)).toarray(),
                                    columns=encoded_col_names)

    # Step 2: Build a BallTree using Hamming distance
    tree = BallTree(encoded_df, metric = distance_metric)

    # Step 3: Query the BallTree to find nearest neighbors
    distances, indices = tree.query(instance_encoded, k=len(df))  # Query all points for sorting

    # Step 4: Ensure diversity in protected features
    diverse_neighborhood_indices = []
    diversity_counts = {feature: 0 for feature in protected_features}

    max_distance = 0
    for idx, indx in enumerate(indices[0]):
      candidate = df.iloc[indx]

      # Add candidate to the neighborhood
      diverse_neighborhood_indices.append(indx) # This leads to S-objective property which is not required here

      max_distance = distances[0, idx]
      # Track diversity for each protected feature
      for feature in protected_features:
        if candidate[feature] != instance.loc[0, feature]:
          # diverse_neighborhood_indices.append(idx)
          diversity_counts[feature] += 1

      # Check if diversity criteria are met
      if all(diversity_counts[feature] >= k for feature in protected_features):
        break  # Stop if we have k diverse values for each protected feature

    # Add elements of the same distance
    # same_distance_indices = indices[0][distances[0]==max_distance]
    # neighborhood_indices = list(set(diverse_neighborhood_indices + list(same_distance_indices)))
    # neighbor_df = df.iloc[neighborhood_indices]
    # max_distance=100
    epsilon = 1e-10
    indices, distances = tree.query_radius(instance_encoded, r=max_distance + epsilon,
                                           return_distance=True, sort_results=True)
    # Making sure that the individual is the first element in the neighbourhood
    indices[0] = np.delete(indices[0], np.where(indices[0]==instance_index)[0][0])
    indices[0] = np.insert(indices[0], 0, instance_index)
    neighbor_df = df.iloc[indices[0]]
    # Step 5: Return the neighborhood in original categorical format
    return neighbor_df


def measure_neighborhood_properties(instance_index, neighborhood, k, protected_features, weakest, method):
    individual = neighborhood.iloc[0, :]
    properties = {}
    vals = [instance_index]
    properties['instance'] = instance_index
    # N-significance
    properties['N-significance'] = k
    vals.append(properties['N-significance'])
    # e-diversity
    for protected_feature in protected_features:
        individual_feature_val = individual[protected_feature]
        prob_individual_feature_val = (
                len(neighborhood[protected_feature][neighborhood[protected_feature] == individual_feature_val])
                / len(neighborhood))
        properties[f'diversity_{protected_feature}'] = 1 - np.abs((2*prob_individual_feature_val)-1)
        vals.append(properties[f'diversity_{protected_feature}'])
        for w in weakest:
            if '&' in w:
                weakest.extend(w.split('&'))
        if f'{protected_feature}=EBA-{individual_feature_val}-total' in weakest if method == methods.Our else f'{protected_feature}={individual_feature_val}' in weakest:
            properties[f'biased_against_{protected_feature}'] = f'Biased against {protected_feature}={individual_feature_val}'
            vals.append(properties[f'biased_against_{protected_feature}'])
        else:
            properties[f'biased_against_{protected_feature}'] = f'Not Biased against {individual_feature_val}'
            vals.append(properties[f'biased_against_{protected_feature}'])
    # S-Objective is satisfied for KNN
    properties['S-Objective'] = 'true'
    vals.append(properties['S-Objective'])
    return properties, vals


def measure_neighborhood_properties_(neighborhood, k, protected_features):
    individual = neighborhood.iloc[0, :]
    properties = {}
    # N-significance
    properties['N-significance'] = k
    # e-diversity
    for protected_feature in protected_features:
        individual_feature_val = individual[protected_feature]
        prob_individual_feature_val = (
                len(neighborhood[protected_feature][neighborhood[protected_feature] == individual_feature_val])
                / len(neighborhood))
        properties[f'diversity_{protected_feature}'] = 1 - np.abs((2*prob_individual_feature_val)-1)
    # S-Objective is satisfied for KNN
    properties['S-Objective'] = 'true'
    return properties





