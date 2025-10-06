import numpy as np
import pandas as pd
import arff
from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage, ACSMobility, ACSTravelTime
from sklearn.metrics import normalized_mutual_info_score
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler, MinMaxScaler
from folktables import generate_categories
from folktables import BasicProblem
from folktables import acs
# from src.Classifier import *
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans

import math


def GermanCredit(n_samples=1000, parameters = [], seed=0):
    names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount',
             'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors',
             'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing',
             'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']
    data = pd.read_csv('datasets/german.data', names=names, delimiter=' ')
    data.classification.replace([1, 2], [1, 0], inplace=True)
    numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age',
               'existingcredits', 'peopleliable', 'classification']
    catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
               'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job',
               'telephone', 'foreignworker']
    numdata_std = pd.DataFrame(StandardScaler().fit_transform(data[numvars].drop(['classification'], axis=1)))

    # Define logical mappings for features with order
    logical_mappings = {
        'existingchecking': {'A11': 0, 'A12': 1, 'A13': 2, 'A14': 3},
        'credithistory': {'A30': 0, 'A31': 1, 'A32': 2, 'A33': 3, 'A34': 4},
        'savings': {'A61': 0, 'A62': 1, 'A63': 2, 'A64': 3, 'A65': 4},
        'employmentsince': {'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4},
        'job': {'A171': 0, 'A172': 1, 'A173': 2, 'A174': 3},
    }

    # Apply logical mappings
    for feature, mapping in logical_mappings.items():
        if feature in data.columns:
            data[feature] = data[feature].map(mapping)

    # For features without logical order, use label encoding
    from sklearn.preprocessing import LabelEncoder

    no_order_features = ['purpose', 'statussex', 'otherdebtors', 'property',
                         'otherinstallmentplans', 'housing', 'telephone', 'foreignworker']

    for feature in no_order_features:
        if feature in data.columns:
            data[feature] = LabelEncoder().fit_transform(data[feature])

    # print()
    # for feature in numvars: # print unique values for each feature
    #     print(feature, data[feature].unique())
    # print()
    # for feature in catvars: # print unique values for each feature
    #     print(feature, data[feature].unique())

    sensitive_attribute = 'age'
    age_threshold = 50
    data[sensitive_attribute] = data[sensitive_attribute].apply(lambda x: 1 if x > age_threshold else 0)
    general_target = 'classification'
    dataframe = data
    features_to_use = [feature for feature in dataframe.columns if
                       feature != sensitive_attribute and feature != general_target]

    # purpose, peopleliable, credithistory, otherinstallmentplans, housing
    features_to_use = ['creditamount', 'purpose', 'peopleliable', 'credithistory', 'otherinstallmentplans', 'housing']

    for feature in dataframe.columns:
        if feature not in features_to_use and feature != general_target and feature != sensitive_attribute:
            dataframe = dataframe.drop(feature, axis=1)

    # print(dataframe.head())

    bins = [len(dataframe[feature].unique()) for feature in features_to_use] + [2, 2]
    bins[0] = 10
    dataframe['creditamount'] = pd.cut(dataframe['creditamount'], bins[0], labels=False)
    bins = tuple(bins)
    sensitive_attribute = [sensitive_attribute]
    output = [general_target]
    # return df, features, sensitive_attribute, output, bins
    return dataframe, features_to_use, sensitive_attribute, output, bins
# test model
def generate_data_model_1(n_samples=10000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)

    # Generate X1 as a function of A
    X1 = (A + np.random.normal(0, 1, n_samples))
    X1 = np.where(X1 > 0, 1, 0)
    # Generate X2 as a function of X1
    X2 = X1 + np.random.normal(0, 1, n_samples)
    X2 = np.where(X2 > 0, 1, 0)
    # Generate X3 as a function of X1 and X2
    X3 = 0.5 * X1 + 0.5 * X2 + np.random.normal(0, 1, n_samples)
    X3 = np.where(X3 > 0, 1, 0)
    # Generate X4 as a function of X3
    X4 = X3 + np.random.normal(0, 1, n_samples)
    X4 = np.where(X4 > 0, 1, 0)
    # Generate X5 as a function of A and X1
    X5 = 0.5 * A + 0.5 * X1 + np.random.normal(0, 1, n_samples)
    X5 = np.where(X5 > 0, 1, 0)
    # Generate Y as a function of X2, X4, and X5
    Y = 0.33 * X2 + 0.33 * X4 + 0.33 * X5 + np.random.normal(0, 1, n_samples)
    Y = np.where(Y > 0, 1, 0)
    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'X5': X5,
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4', 'X5']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (2, 2, 2, 2, 2, 2, 2)
    return df, features, sensitive_attribute, output, bins

# Gaussian models
def generate_data_model_Gaussian(n_samples=10000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)

    # Generate X1 as a function of A
    X1 = (A + np.random.normal(0, 1, n_samples))
    # X1 = np.where(X1 > 0, 1, 0)
    # Generate X2 as a function of X1
    X2 = X1 + np.random.normal(0, 1, n_samples)
    # X2 = np.where(X2 > 0, 1, 0)
    # X3 = np.where(X3 > 0, 1, 0)
    # Generate X4 as a function of X3
    X4 = np.random.normal(0, 1, n_samples)

    # Generate X3 as a function of X1 and X2
    X3 = X1 + X2 + X4 + np.random.normal(0, 1, n_samples)
    # X4 = np.where(X4 > 0, 1, 0)
    # Generate X5 as a function of A and X1
    X5 = A + X1 + np.random.normal(0, 1, n_samples)
    # X5 = np.where(X5 > 0, 1, 0)
    # Generate Y as a function of X2, X4, and X5
    Y = X3 + X4 + X5 + np.random.normal(0, 1, n_samples)
    Y = np.where(Y > 0, 1, 0)
    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'X5': X5,
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4', 'X5']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (10, 10, 10, 10, 10, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_data_model_Gaussian_graph_new_1(n_samples=10000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)

    X4 = np.random.normal(0, 1, n_samples)
    X5 = np.random.normal(0, 1, n_samples)

    X2 = A + X4 + np.random.normal(0, 1, n_samples)
    X3 = A + X5 + np.random.normal(0, 1, n_samples)

    X1 = A + X4 + X5 + np.random.normal(0, 1, n_samples)

    Y = X5 + X4 + np.random.normal(0, 1, n_samples)
    Y = np.where(Y > 0, 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1/np.std(X1),
        'X2': X2/np.std(X2),
        'X3': X3/np.std(X3),
        'X4': X4/np.std(X4),
        'X5': X5/np.std(X5),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4', 'X5']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (10, 10, 10, 10, 10, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_data_model_Gaussian_graph_new_2(n_samples=10000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)

    # Generate X1 as a function of A
    X1 = (4*A + np.random.normal(0, 1, n_samples))
    # X1 = np.where(X1 > 0, 1, 0)
    # Generate X2 as a function of X1
    X2 = 4*X1 + np.random.normal(0, 1, n_samples)
    # X2 = np.where(X2 > 0, 1, 0)
    # X3 = np.where(X3 > 0, 1, 0)
    # Generate X4 as a function of X3
    X4 = np.random.normal(0, 1, n_samples)

    # Generate X3 as a function of X1 and X2
    X3 = 4*X1 + X2 + 2*X4 + np.random.normal(0, 1, n_samples)
    # X4 = np.where(X4 > 0, 1, 0)
    # Generate X5 as a function of A and X1
    X5 = A + X1 + np.random.normal(0, 1, n_samples)
    # X5 = np.where(X5 > 0, 1, 0)
    # Generate Y as a function of X2, X4, and X5
    Y = 4*X3 + 2*X4 + 0.5*X5 + np.random.normal(0, 1, n_samples)
    Y = np.where(Y > 0, 1, 0)
    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'X5': X5 / np.std(X5),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4', 'X5']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (10, 10, 10, 10, 10, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_data_model_Gaussian_graph_new_3(n_samples=10000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)

    X1 = 1*A + np.random.normal(0, 1, n_samples)
    X2 = 2*A + np.random.normal(0, 1, n_samples)
    X3 = 3*A + np.random.normal(0, 1, n_samples)

    X4 = 3*X1 + np.random.normal(0, 1, n_samples)
    X5 = 3*X2 + np.random.normal(0, 1, n_samples)
    X6 = 3*X3 + np.random.normal(0, 1, n_samples)

    Y = 2*X3 + 2*X4 + 2*X5 + np.random.normal(0, 1, n_samples)
    Y = np.where(Y > 0, 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'X5': X5 / np.std(X5),
        'X6': X6 / np.std(X6),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (10, 10, 10, 10, 10, 10, 2, 2)
    return df, features, sensitive_attribute, output, bins

# models used for Benchmarking experiment
def generate_data_model_simple(n_samples=10000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)

    # Generate X1 as a function of A
    X1 = (parameters[0]*A + np.random.normal(0, 0.9, n_samples))
    X1 = np.where(X1 > 0, 1, 0)
    # Generate X2 as a function of X1
    X2 = parameters[1]*A +X1 + np.random.normal(0, 0.9, n_samples)
    X2 = np.where(X2 > 0, 1, 0)
    # Generate X3 as a function of X1 and X2
    X3 = parameters[2]*A +0.5 * X1 + 0.5 * X2 + np.random.normal(0, 0.9, n_samples)
    X3 = np.where(X3 > 0, 1, 0)
    # Generate X4 as a function of X3
    X4 = parameters[3]*A +X3 + np.random.normal(0, 0.9, n_samples)
    X4 = np.where(X4 > 0, 1, 0)
    # Generate X5 as a function of A and X1
    X5 =parameters[4]*A + 0.5 * X1 + np.random.normal(0, 0.9, n_samples)
    X5 = np.where(X5 > 0, 1, 0)
    # Generate Y as a function of X2, X4, and X5
    Y = 0.33 * X2 + 0.33 * X4 + 0.33 * X5 + np.random.normal(0, 0.9, n_samples)
    Y = np.where(Y > 0, 1, 0)
    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'X5': X5,
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4', 'X5']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (2, 2, 2, 2, 2, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_data_model_parameterized(n_samples=10000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, min(max(parameters[0], 0.3), 0.7), n_samples)

    # Generate X1 as a function of A
    X1 = (parameters[1] * A + np.random.normal(-0.5, min(max(parameters[2], 0.25), 0.75), n_samples))
    X1 = np.where(X1 > 0, 1, 0)
    # Generate X2 as a function of X1
    X2 = X1 + np.random.normal(-0.5, min(max(parameters[3], 0.25), 0.75), n_samples)
    X2 = np.where(X2 > 0, 1, 0)
    # Generate X3 as a function of X1 and X2
    X3 = parameters[4] * X1 + parameters[5] * X2 + np.random.normal(-0.5,
                                                                            min(max(parameters[6], 0.25), 0.75),
                                                                            n_samples)
    X3 = np.where(X3 > 0, 1, 0)
    # Generate X4 as a function of X3
    X4 = X3 + np.random.normal(-0.5, min(max(parameters[7], 0.25), 1.25), n_samples)
    X4 = np.where(X4 > 0, 1, 0)
    # Generate X5 as a function of A and X1
    X5 = parameters[8] * A + parameters[9] * X1 + np.random.normal(-0.5,
                                                                           min(max(parameters[10], 0.25), 0.75),
                                                                           n_samples)
    X5 = np.where(X5 > 0, 1, 0)
    # Generate Y as a function of X2, X4, and X5
    Y = parameters[11] * X2 + parameters[12] * X4 + parameters[13] * X5 + np.random.normal(-0.5, min(max(
        parameters[14], 0.25), 0.75), n_samples)
    Y = np.where(Y > 0, 1, 0)
    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'X5': X5,
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4', 'X5']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (2, 2, 2, 2, 2, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_data_model_parameterized_complex(n_samples=10000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    # 0-9 bernoulli
    # 10-25 gaussian
    # 26-32 uniform

    A = np.random.binomial(1, min(max(parameters[26], 0.3), 0.7), n_samples)

    # Generate X1 as a function of A
    X1 = parameters[10]*parameters[0] * A + np.random.normal(0, min(max(parameters[27], 0.25), 0.75), n_samples)
    X1 = np.where(X1 > 0, 1, 0)
    # Generate X2 as a function of X1
    X2 = parameters[11]*parameters[1] * A + parameters[15] * X1 + np.random.normal(0, min(max(parameters[28], 0.25), 0.75), n_samples)
    X2 = np.where(X2 > 0, 1, 0)
    # Generate X3 as a function of X1 and X2
    X3 = parameters[12]*parameters[2] * A + parameters[16] * X1 + parameters[17] * X2 + np.random.normal(0,
                                                                            min(max(parameters[29], 0.25), 0.75),
                                                                            n_samples)
    X3 = np.where(X3 > 0, 1, 0)
    # Generate X4 as a function of X3
    X4 = parameters[13]*parameters[3] * A + parameters[18]*X3 + np.random.normal(0, min(max(parameters[30], 0.25), 1.25), n_samples)
    X4 = np.where(X4 > 0, 1, 0)
    # Generate X5 as a function of A and X1
    X5 = parameters[14]*parameters[4] * A + parameters[19] * X1 + np.random.normal(0,
                                                                           min(max(parameters[31], 0.25), 0.75),
                                                                           n_samples)
    X5 = np.where(X5 > 0, 1, 0)
    # Generate Y as a function of X2, X4, and X5
    Y = parameters[20]*parameters[5]* X1 + parameters[21]*parameters[6] * parameters[22]*X2+ parameters[23]*parameters[7] * X3 + parameters[24]*parameters[8] * X4 + parameters[25]*parameters[9] * X5 + np.random.normal(0, min(max(
        parameters[32], 0.25), 0.75), n_samples)
    Y = np.where(Y > 0, 1, 0)
    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'X5': X5,
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4', 'X5']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (2, 2, 2, 2, 2, 2, 2)
    return df, features, sensitive_attribute, output, bins

# Repeating the results of the ProPublicaCOMPASDataset
def generate_data_model_ProPublicaCOMPASDataset(n_samples=10000, parameters=[], seed=0):
    # Load the data
    data = pd.read_csv('datasets/compas-scores-two-years.csv')

    features_to_keep = ['age', 'c_charge_degree', 'sex', 'priors_count.1', 'c_jail_in', 'c_jail_out', 'race',
                        'two_year_recid']
    # Filter the data
    filtered_data = data.filter(features_to_keep)
    for column in filtered_data.columns:
        print(f"Column: {column}")
        print(f"Unique Values: {filtered_data[column].unique()}")
        print(f"Value Counts:\n{filtered_data[column].value_counts()}")
        print("\n" + "#" * 40 + "\n")
    # Print the header of the filtered data
    filtered_data = filtered_data.dropna()

    # print(filtered_data.head())

    # print summary of the data for race column


    races_to_keep = ['African-American', 'Caucasian']
    # Filter the data
    filtered_data = filtered_data[filtered_data['race'].isin(races_to_keep)]

    # Convert 'c_jail_in' and 'c_jail_out' to datetime
    filtered_data['c_jail_in'] = pd.to_datetime(filtered_data['c_jail_in'])
    filtered_data['c_jail_out'] = pd.to_datetime(filtered_data['c_jail_out'])

    # Create 'Length Of Stay' column
    filtered_data['Length Of Stay'] = (filtered_data['c_jail_out'] - filtered_data['c_jail_in']).dt.days
    filtered_data.drop(['c_jail_in', 'c_jail_out'], axis=1, inplace=True)

    filtered_data['age'] = pd.cut(filtered_data['age'], bins=[0, 24, 44, np.inf], labels=[0, 1, 2])

    # Transform 'c_charge_degree' column
    filtered_data['c_charge_degree'] = filtered_data['c_charge_degree'].map({'M': 0, 'F': 1})

    # Transform 'sex' column
    filtered_data['sex'] = filtered_data['sex'].map({'Male': 0, 'Female': 1})

    # Transform 'priors_count.1' column
    filtered_data['priors_count.1'] = pd.cut(filtered_data['priors_count.1'], bins=[-1, 0.5, 3.5, np.inf],
                                             labels=[0, 1, 2])

    # Transform 'Length Of Stay' column
    filtered_data['Length Of Stay'] = pd.cut(filtered_data['Length Of Stay'], bins=[-1, 7.5, 90.5, np.inf],
                                             labels=[0, 1, 2])

    filtered_data['race'] = filtered_data['race'].map({'African-American': 0, 'Caucasian': 1})

    # Print the header of the transformed data
    # print(filtered_data.head())

    features = ['age', 'c_charge_degree', 'sex', 'priors_count.1', 'race', 'two_year_recid', 'Length Of Stay']
    # Define the new order of columns
    new_order = ['age', 'c_charge_degree', 'sex', 'priors_count.1', 'Length Of Stay', 'race', 'two_year_recid']

    # Reorder the columns
    filtered_data = filtered_data.reindex(columns=new_order)
    # print(filtered_data.head())
    features = ['age', 'c_charge_degree', 'sex', 'priors_count.1', 'Length Of Stay']
    sensitive_attribute = ['race']
    output = ['two_year_recid']
    filtered_data = filtered_data.dropna()
    filtered_data['age'] = filtered_data['age'].astype(int)
    filtered_data['priors_count.1'] = filtered_data['priors_count.1'].astype(int)
    filtered_data['Length Of Stay'] = filtered_data['Length Of Stay'].astype(int)

    bins = (3, 2, 2, 3, 3, 2, 2)
    # print(filtered_data.head())
    # print(features)
    # print(sensitive_attribute)
    # print(output)
    # print(bins)
    # print(filtered_data.dtypes)
    # print(len(filtered_data))


    return filtered_data, features, sensitive_attribute, output, bins

def generate_data_model_ProPublicaCOMPASDataset_enhanced_preprocessing(n_samples=10000, parameters=[], seed=0):
    data = pd.read_csv('datasets/compas-scores-two-years.csv')
    # print(data.isnull().sum())

    # filtered_data = data[(data['days_b_screening_arrest'] <= 30) &
    #                      (data['days_b_screening_arrest'] >= -30) &
    #                      (data['is_recid'] != -1) &
    #                      (data['c_charge_degree'] != 'O') &
    #                      (data['score_text'] != 'N/A')]

    filtered_data = data[(data['days_b_screening_arrest'] <= 30) &
                         (data['days_b_screening_arrest'] >= -30) &
                         (data['is_recid'] != -1) &
                         (data['c_charge_degree'] != 'O') &
                         (data['score_text'] != 'N/A')]

    df = filtered_data[
        ['age_cat', 'c_charge_degree', 'sex', 'priors_count', 'c_jail_in', 'c_jail_out', 'race', 'two_year_recid']]
    races_to_keep = ['African-American', 'Caucasian']
    df = df[df['race'].isin(races_to_keep)]

    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])

    # Create 'Length Of Stay' column
    df['Length Of Stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df.drop(['c_jail_in', 'c_jail_out'], axis=1, inplace=True)

    df['age_cat'] = df['age_cat'].map({'Less than 25': 0, '25 - 45': 1, 'Greater than 45': 2})
    df['c_charge_degree'] = df['c_charge_degree'].map({'M': 0, 'F': 1})
    df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
    # # plot histogram of 'priors_count' column
    # df['priors_count'].hist(bins = 20)
    # plt.show()
    # # plot histogram of 'Length Of Stay' column
    # df['Length Of Stay'].hist(bins = 20)
    # plt.show()
    df['priors_count'] = pd.cut(df['priors_count'], bins=[-1, 0.5, 3.5, np.inf], labels=[0, 1, 2])

    # Transform 'Length Of Stay' column
    df['Length Of Stay'] = pd.cut(df['Length Of Stay'], bins=[-1, 7.5, 90.5, np.inf], labels=[0, 1, 2])

    df['race'] = df['race'].map({'African-American': 0, 'Caucasian': 1})

    new_order = ['age_cat', 'c_charge_degree', 'sex', 'priors_count', 'Length Of Stay', 'race', 'two_year_recid']
    df = df.reindex(columns=new_order)
    # print(len(df))

    features = ['age_cat', 'c_charge_degree', 'sex', 'priors_count', 'Length Of Stay']
    sensitive_attribute = ['race']
    output = ['two_year_recid']
    # check of missing values
    # print(df.isnull().sum())

    df = df.dropna()

    # print(df["race"].value_counts())

    df['age_cat'] = df['age_cat'].astype(int)
    df['priors_count'] = df['priors_count'].astype(int)
    df['Length Of Stay'] = df['Length Of Stay'].astype(int)

    bins = (3, 2, 2, 3, 3, 2, 2)


    return df, features, sensitive_attribute, output, bins


def generate_data_model_ProPublicaCOMPASDataset_enhanced_preprocessing_updated(n_samples=10000, parameters=[], seed=0):
    data = pd.read_csv('datasets/compas-scores-two-years.csv')
    # print(data.isnull().sum())

    # filtered_data = data[(data['days_b_screening_arrest'] <= 30) &
    #                      (data['days_b_screening_arrest'] >= -30) &
    #                      (data['is_recid'] != -1) &
    #                      (data['c_charge_degree'] != 'O') &
    #                      (data['score_text'] != 'N/A')]

    filtered_data = data[(data['days_b_screening_arrest'] <= 30) &
                         (data['days_b_screening_arrest'] >= -30) &
                         (data['is_recid'] != -1) &
                         (data['c_charge_degree'] != 'O') &
                         (data['score_text'] != 'N/A')]

    df = filtered_data[
        ['age_cat', 'c_charge_degree', 'sex', 'priors_count', 'c_jail_in', 'c_jail_out', 'race', 'two_year_recid']]
    races_to_keep = ['African-American', 'Caucasian']
    df = df[df['race'].isin(races_to_keep)]

    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])

    # Create 'Length Of Stay' column
    df['Length Of Stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df.drop(['c_jail_in', 'c_jail_out'], axis=1, inplace=True)

    df['age_cat'] = df['age_cat'].map({'Less than 25': 0, '25 - 45': 1, 'Greater than 45': 2})
    df['c_charge_degree'] = df['c_charge_degree'].map({'M': 0, 'F': 1})
    df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
    # # plot histogram of 'priors_count' column
    # df['priors_count'].hist(bins = 20)
    # plt.show()
    # # plot histogram of 'Length Of Stay' column
    # df['Length Of Stay'].hist(bins = 20)
    # plt.show()
    # df['priors_count'] = pd.cut(df['priors_count'], bins=[-1, 0.5, 3.5, np.inf], labels=[0, 1, 2])
    df['priors_count'] = pd.qcut(df['priors_count'], q=3, labels=False)
    # Transform 'Length Of Stay' column
    # df['Length Of Stay'] = pd.cut(df['Length Of Stay'], bins=[-1, 7.5, 90.5, np.inf], labels=[0, 1, 2])
    df['Length Of Stay'] = pd.qcut(df['Length Of Stay'], q=4, labels=False)
    df['race'] = df['race'].map({'African-American': 0, 'Caucasian': 1})

    new_order = ['age_cat', 'c_charge_degree', 'sex', 'priors_count', 'Length Of Stay', 'race', 'two_year_recid']
    df = df.reindex(columns=new_order)
    # print(len(df))

    features = ['age_cat', 'c_charge_degree', 'sex', 'priors_count', 'Length Of Stay']
    sensitive_attribute = ['race']
    output = ['two_year_recid']
    # check of missing values
    # print(df.isnull().sum())

    df = df.dropna()

    # print(df["race"].value_counts())

    df['age_cat'] = df['age_cat'].astype(int)
    df['priors_count'] = df['priors_count'].astype(int)
    df['Length Of Stay'] = df['Length Of Stay'].astype(int)

    bins = (3, 2, 2, 3, 4, 2, 2)

    categorical_features = ['age_cat', 'c_charge_degree', 'sex',]
    return df, features, sensitive_attribute, output, bins, categorical_features

# Data for Dutta experiment
def generate_data_model_Dutta(n_samples=10000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)

    Z = np.random.binomial(1, 0.5, n_samples)

    U1 = np.random.binomial(1, 0.9, n_samples)

    X1 = U1

    U2 = np.random.binomial(1, 0.5, n_samples)
    X2 = Z + U2

    X3= U2
    N= np.random.binomial(1, 0.1, n_samples)
    Y= Z + U1 + 2* U2 +N

    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'A': Z,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (2, 3, 2, 2, 6)
    return df, features, sensitive_attribute, output, bins

# Gaussian graphs discretized
def generate_data_model_Gaussian_graph_new_1_dis(n_samples=10000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    gamma = parameters[2]
    A = np.random.binomial(1, parameters[1], n_samples)

    X4 = np.random.normal(0, gamma, n_samples)
    X5 = np.random.normal(0, gamma, n_samples)

    X2 = A + X4 + np.random.normal(0, gamma, n_samples)
    X3 = A + X5 + np.random.normal(0, gamma, n_samples)

    X1 = A + X4 + X5 + np.random.normal(0, gamma, n_samples)

    Y = X5 + X4 + np.random.normal(0, gamma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1/np.std(X1),
        'X2': X2/np.std(X2),
        'X3': X3/np.std(X3),
        'X4': X4/np.std(X4),
        'X5': X5/np.std(X5),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4', 'X5']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=5, labels=False)
    # print(df.head(10))
    bins = (5, 5, 5, 5, 5, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_data_model_Gaussian_graph_new_2_dis(n_samples=10000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    gamma = parameters[2]
    # Generate X1 as a function of A
    X1 = ((4*A-2) + np.random.normal(0, gamma, n_samples))
    # X1 = np.where(X1 > 0, 1, 0)
    # Generate X2 as a function of X1
    X2 = 4*X1 + np.random.normal(0, gamma, n_samples)
    # X2 = np.where(X2 > 0, 1, 0)
    # X3 = np.where(X3 > 0, 1, 0)
    # Generate X4 as a function of X3
    X4 = np.random.normal(0, gamma, n_samples)

    # Generate X3 as a function of X1 and X2
    X3 = 4*X1 + X2 + 2*X4 + np.random.normal(0, gamma, n_samples)
    # X4 = np.where(X4 > 0, 1, 0)
    # Generate X5 as a function of A and X1
    X5 = (A-0.5) + X1 + np.random.normal(0, gamma, n_samples)
    # X5 = np.where(X5 > 0, 1, 0)
    # Generate Y as a function of X2, X4, and X5
    Y = 4*X3 + 2*X4 + 0.5*X5 + np.random.normal(0, gamma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)
    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'X5': X5 / np.std(X5),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4', 'X5']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=5, labels=False)
    bins = (5, 5, 5, 5, 5, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_data_model_Gaussian_graph_new_3_dis(n_samples=10000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)
    gamma = 0.5
    X1 = 1*A-0.5 + np.random.normal(0, gamma, n_samples)
    X2 = 2*A-1 + np.random.normal(0, gamma, n_samples)
    X3 = 3*A-1.5 + np.random.normal(0, gamma, n_samples)

    X4 = 3*X1 + np.random.normal(0, gamma, n_samples)
    X5 = 3*X2 + np.random.normal(0, gamma, n_samples)
    X6 = 3*X3 + np.random.normal(0, gamma, n_samples)

    Y = 2*X4 + 2*X5 + 2*X6 + np.random.normal(0, gamma, n_samples)
    Y = np.where(Y > 0, 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'X5': X5 / np.std(X5),
        'X6': X6 / np.std(X6),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=parameters[0], labels=False)
    bins = (parameters[0], parameters[0], parameters[0], parameters[0], parameters[0], parameters[0], 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_data_model_Gaussian_graph_new_3_red_dis(n_samples=10000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    gamma = parameters[2]
    X1 = 1*A-0.5 + np.random.normal(0, gamma, n_samples)
    X2 = 3*A-1.5 + np.random.normal(0, gamma, n_samples)

    X3 = 3*X1 + np.random.normal(0, gamma, n_samples)
    X4 = 3*X2 + np.random.normal(0, gamma, n_samples)

    Y = 2*X4 + 2*X3 + np.random.normal(0, gamma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_data_model_Gaussian_graph_new_1_modified_dis(n_samples=10000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)

    gamma = parameters[2]
    X4 = np.random.normal(0, gamma, n_samples)
    X5 = np.random.normal(0, gamma, n_samples)

    X2 = (2*A-1) + X4 + np.random.normal(0, gamma, n_samples)
    X3 = (2*A-1) + X5 + np.random.normal(0, gamma, n_samples)

    X1 = (2*A-1) + X4 + X5 + np.random.normal(0, gamma, n_samples)

    Y = X1 + X2 + X3 + X5 + X4 + np.random.normal(0, gamma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1/np.std(X1),
        'X2': X2/np.std(X2),
        'X3': X3/np.std(X3),
        'X4': X4/np.std(X4),
        'X5': X5/np.std(X5),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4', 'X5']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=5, labels=False)
    bins = (5, 5, 5, 5, 5, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_data_model_Gaussian_graph_new_4_dis(n_samples=10000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    gamma = parameters[2]
    X1 = A + np.random.normal(0, gamma, n_samples)
    X3 = A + np.random.normal(0, gamma, n_samples)
    X2 = X1 - X3 + np.random.normal(0, gamma, n_samples)

    Y = X1 + 2*X2 + X3 + np.random.normal(0, gamma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def adult_data(n_samples=10000, parameters = [], seed=0):
    # ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'relationship']
    # max [10, 10, 10, 10, 10, 6]

    dataframe = pd.read_csv('datasets/adult.csv')

    # print()
    # print("Number of Rows",data.shape[0])
    # print("Number of Columns",data.shape[1])

    dataframe['workclass'] = dataframe['workclass'].replace('?', np.nan)
    dataframe['occupation'] = dataframe['occupation'].replace('?', np.nan)
    dataframe['native-country'] = dataframe['native-country'].replace('?', np.nan)
    dataframe.dropna(how='any', inplace=True)

    dataframe = dataframe.drop_duplicates()
    # print()
    # print("Number of Rows",data.shape[0])
    # print("Number of Columns",data.shape[1])
    features = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                'marital-status', 'occupation', 'relationship', 'race', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country']

    sensitive_attribute = 'gender'
    general_target = 'income'

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "native-country",
    ]
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x == '>50K' else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 'Female' else 0)

    # Convert categorical features to numerical indices
    for feature in categorical_features:
        dataframe[feature] = pd.factorize(dataframe[feature])[0]

    # Print the dataframe to confirm conversion
    # print(dataframe.head())

    features_to_keep = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'relationship']
    for feature in features:
        if feature not in features_to_keep and feature != sensitive_attribute and feature != general_target:
            dataframe = dataframe.drop(feature, axis=1)

    # plot the histograms of the age, educational-num, capital-loss', 'hours-per-week'
    # dataframe['age'].hist(bins=20)
    # plt.title('Age')
    # plt.show()
    # dataframe['educational-num'].hist(bins=20)
    # plt.title('Educational-num')
    # plt.show()
    # dataframe['capital-gain'].hist(bins=20)
    # plt.title('Capital-gain')
    # plt.show()
    # dataframe['capital-loss'].hist(bins=20)
    # plt.title('Capital-loss')
    # plt.show()
    # dataframe['hours-per-week'].hist(bins=20)
    # plt.title('Hours-per-week')
    # plt.show()


    # apply bining to ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'relationship']
    dataframe['age'] = pd.cut(dataframe['age'], bins=parameters[0], labels=False)
    dataframe['educational-num'] = pd.cut(dataframe['educational-num'], bins=parameters[1], labels=False)
    dataframe['capital-gain'] = pd.cut(dataframe['capital-gain'], bins=parameters[2], labels=False)
    dataframe['capital-loss'] = pd.cut(dataframe['capital-loss'], bins=parameters[3], labels=False)
    dataframe['hours-per-week'] = pd.cut(dataframe['hours-per-week'], bins=parameters[4], labels=False)
    dataframe['relationship'] = pd.cut(dataframe['relationship'], bins=parameters[5], labels=False)

    features = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'relationship']
    bins = (parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], 2, 2)
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]

    return dataframe, features, sensitive_attribute, output, bins


def adult_data_updated2(n_samples=10000, parameters = [], seed=0, quantization_type = 0):
    # ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'relationship']
    # max [10, 10, 10, 10, 10, 6]

    dataframe = pd.read_csv('datasets/adult.csv')

    # print()
    # print("Number of Rows",data.shape[0])
    # print("Number of Columns",data.shape[1])

    dataframe['workclass'] = dataframe['workclass'].replace('?', np.nan)
    dataframe['occupation'] = dataframe['occupation'].replace('?', np.nan)
    dataframe['native-country'] = dataframe['native-country'].replace('?', np.nan)
    dataframe.dropna(how='any', inplace=True)

    dataframe = dataframe.drop_duplicates()
    # print()
    # print("Number of Rows",data.shape[0])
    # print("Number of Columns",data.shape[1])
    features = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                'marital-status', 'occupation', 'relationship', 'race', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country']

    sensitive_attribute = 'gender'
    general_target = 'income'

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "native-country",
    ]
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x == '>50K' else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 'Female' else 0)

    # Convert categorical features to numerical indices
    for feature in categorical_features:
        dataframe[feature] = pd.factorize(dataframe[feature])[0]

    # Print the dataframe to confirm conversion
    # print(dataframe.head())

    features_to_keep = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'relationship']
    for feature in features:
        if feature not in features_to_keep and feature != sensitive_attribute and feature != general_target:
            dataframe = dataframe.drop(feature, axis=1)

    # plot the histograms of the age, educational-num, capital-loss', 'hours-per-week'
    # dataframe['age'].hist(bins=20)
    # plt.title('Age')
    # plt.show()
    # dataframe['educational-num'].hist(bins=20)
    # plt.title('Educational-num')
    # plt.show()
    # dataframe['capital-gain'].hist(bins=20)
    # plt.title('Capital-gain')
    # plt.show()
    # dataframe['capital-loss'].hist(bins=20)
    # plt.title('Capital-loss')
    # plt.show()
    # dataframe['hours-per-week'].hist(bins=20)
    # plt.title('Hours-per-week')
    # plt.show()


    # apply bining to ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'relationship']
    if quantization_type == 0:
        dataframe['age'] = pd.cut(dataframe['age'], bins=parameters[0], labels=False)
    else:
        dataframe['age'] = pd.qcut(dataframe['age'], q=parameters[0], labels=False)
    if quantization_type == 0:
        dataframe['educational-num'] = pd.cut(dataframe['educational-num'], bins=parameters[1], labels=False)
    else:
        dataframe['educational-num'] = pd.qcut(dataframe['educational-num'], q=parameters[1], labels=False)
    dataframe['capital-gain'] = pd.cut(dataframe['capital-gain'], bins=parameters[2], labels=False)
    dataframe['capital-loss'] = pd.cut(dataframe['capital-loss'], bins=parameters[3], labels=False)
    if quantization_type == 0:
        dataframe['hours-per-week'] = pd.cut(dataframe['hours-per-week'], bins=parameters[4], labels=False)
    else:
        dataframe['hours-per-week'] = pd.qcut(dataframe['hours-per-week'], q=parameters[4], labels=False)
    dataframe['relationship'] = pd.cut(dataframe['relationship'], bins=parameters[5], labels=False)

    features = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'relationship']
    bins = (parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], 2, 2)
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]

    return dataframe, features, sensitive_attribute, output, bins


def adult_data_updated(n_samples=10000, parameters = [], seed=0):
    # ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'relationship']
    # max [10, 10, 10, 10, 10, 6]

    dataframe = pd.read_csv('datasets/adult.csv')

    # print()
    # print("Number of Rows",data.shape[0])
    # print("Number of Columns",data.shape[1])

    dataframe['workclass'] = dataframe['workclass'].replace('?', np.nan)
    dataframe['occupation'] = dataframe['occupation'].replace('?', np.nan)
    dataframe['native-country'] = dataframe['native-country'].replace('?', np.nan)
    dataframe.dropna(how='any', inplace=True)

    dataframe = dataframe.drop_duplicates()
    # print()
    # print("Number of Rows",data.shape[0])
    # print("Number of Columns",data.shape[1])
    features = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                'marital-status', 'occupation', 'relationship', 'race', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country']

    sensitive_attribute = 'gender'
    general_target = 'income'

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "native-country",
    ]
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x == '>50K' else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 'Female' else 0)

    # Convert categorical features to numerical indices
    for feature in categorical_features:
        dataframe[feature] = pd.factorize(dataframe[feature])[0]

    # Print the dataframe to confirm conversion
    # print(dataframe.head())

    features_to_keep = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'relationship']
    for feature in features:
        if feature not in features_to_keep and feature != sensitive_attribute and feature != general_target:
            dataframe = dataframe.drop(feature, axis=1)

    # plot the histograms of the age, educational-num, capital-loss', 'hours-per-week'
    # dataframe['age'].hist(bins=20)
    # plt.title('Age')
    # plt.show()
    # dataframe['educational-num'].hist(bins=20)
    # plt.title('Educational-num')
    # plt.show()
    # dataframe['capital-gain'].hist(bins=20)
    # plt.title('Capital-gain')
    # plt.show()
    # dataframe['capital-loss'].hist(bins=20)
    # plt.title('Capital-loss')
    # plt.show()
    # dataframe['hours-per-week'].hist(bins=20)
    # plt.title('Hours-per-week')
    # plt.show()


    # apply bining to ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'relationship']
    dataframe['age'] = pd.qcut(dataframe['age'], q=10, labels=False)
    dataframe['educational-num'] = pd.cut(dataframe['educational-num'], bins=10, labels=False)
    dataframe['capital-gain'] = pd.cut(dataframe['capital-gain'], bins=[-1, 500, 100000], labels=False)
    dataframe['capital-loss'] = pd.cut(dataframe['capital-loss'], bins=[-1, 1000, 1000000], labels=False)
    dataframe['hours-per-week'] = pd.cut(dataframe['hours-per-week'], bins=10, labels=False)
    # dataframe['relationship'] = pd.cut(dataframe['relationship'], bins=parameters[5], labels=False)

    features = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'relationship']
    bins = (dataframe['age'].nunique(), dataframe['educational-num'].nunique(), dataframe['capital-gain'].nunique(),
            dataframe['capital-loss'].nunique(), dataframe['hours-per-week'].nunique(), dataframe['relationship'].nunique(), 2, 2)
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]

    return dataframe, features, sensitive_attribute, output, bins, ['relationship']

def adult_data2(n_samples=10000, parameters = [], seed=0):
    # ['fnlwgt', 'hours-per-week', 'relationship', 'marital-status', 'occupation']
    # max [10, 10, 6, 7, 10]

    dataframe = pd.read_csv('datasets/adult.csv')

    # print()
    # print("Number of Rows",data.shape[0])
    # print("Number of Columns",data.shape[1])

    dataframe['workclass'] = dataframe['workclass'].replace('?', np.nan)
    dataframe['occupation'] = dataframe['occupation'].replace('?', np.nan)
    dataframe['native-country'] = dataframe['native-country'].replace('?', np.nan)
    dataframe.dropna(how='any', inplace=True)

    dataframe = dataframe.drop_duplicates()
    # print()
    # print("Number of Rows",data.shape[0])
    # print("Number of Columns",data.shape[1])
    features = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                'marital-status', 'occupation', 'relationship', 'race', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country']

    sensitive_attribute = 'gender'
    general_target = 'income'

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "native-country",
    ]
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x == '>50K' else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 'Female' else 0)

    # Convert categorical features to numerical indices
    for feature in categorical_features:
        dataframe[feature] = pd.factorize(dataframe[feature])[0]

    # Print the dataframe to confirm conversion
    # print(dataframe.head())

    features_to_keep = ['fnlwgt', 'hours-per-week', 'relationship', 'marital-status', 'occupation']
    for feature in features:
        if feature not in features_to_keep and feature != sensitive_attribute and feature != general_target:
            dataframe = dataframe.drop(feature, axis=1)

    # occupation
    dataframe['occupation'] = dataframe['occupation'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['occupation_encoded'] = target_encoder.fit_transform(dataframe['occupation'], dataframe[general_target])
    occ_index = dataframe.columns.get_loc('occupation')
    dataframe.drop('occupation', axis=1, inplace=True)
    # dataframe.insert(occ_index, 'occupation_encoded', dataframe.pop('occupation'))
    # rename the column
    dataframe.rename(columns={'occupation_encoded': 'occupation'}, inplace=True)

    # features_to_keep = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'relationship']
    # for feature in features:
    #     if feature not in features_to_keep and feature != sensitive_attribute and feature != general_target:
    #         dataframe = dataframe.drop(feature, axis=1)

    # apply bining to ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'relationship']
    dataframe['fnlwgt'] = pd.cut(dataframe['fnlwgt'], bins=parameters[0], labels=False)
    dataframe['hours-per-week'] = pd.cut(dataframe['hours-per-week'], bins=parameters[1], labels=False)
    dataframe['relationship'] = pd.cut(dataframe['relationship'], bins=parameters[2], labels=False)
    dataframe['marital-status'] = pd.cut(dataframe['marital-status'], bins=parameters[3], labels=False)
    dataframe['occupation'] = pd.cut(dataframe['occupation'], bins=parameters[4], labels=False)

    features =  ['fnlwgt', 'hours-per-week', 'relationship', 'marital-status', 'occupation'] # [10, 10, 6, 7, 10]
    bins = (parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], 2, 2)
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]

    return dataframe, features, sensitive_attribute, output, bins

def adult_data3(n_samples=10000, parameters = [], seed=0):
    # ['fnlwgt', 'hours-per-week', 'relationship', 'marital-status', 'occupation']
    # max [10, 10, 6, 7, 10]

    dataframe = pd.read_csv('datasets/adult.csv')

    # print()
    # print("Number of Rows",data.shape[0])
    # print("Number of Columns",data.shape[1])

    dataframe['workclass'] = dataframe['workclass'].replace('?', np.nan)
    dataframe['occupation'] = dataframe['occupation'].replace('?', np.nan)
    dataframe['native-country'] = dataframe['native-country'].replace('?', np.nan)
    dataframe.dropna(how='any', inplace=True)

    dataframe = dataframe.drop_duplicates()
    # print()
    # print("Number of Rows",data.shape[0])
    # print("Number of Columns",data.shape[1])
    features = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                'marital-status', 'occupation', 'relationship', 'race', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country']

    sensitive_attribute = 'gender'
    general_target = 'income'

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "native-country",
    ]
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x == '>50K' else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 'Female' else 0)

    # Convert categorical features to numerical indices
    for feature in categorical_features:
        dataframe[feature] = pd.factorize(dataframe[feature])[0]

    # Print the dataframe to confirm conversion
    # print(dataframe.head())

    features_to_keep = ['age', 'hours-per-week', 'capital-gain', 'capital-loss']
    for feature in features:
        if feature not in features_to_keep and feature != sensitive_attribute and feature != general_target:
            dataframe = dataframe.drop(feature, axis=1)

    for i in range(len(features_to_keep)):
        dataframe[features_to_keep[i]] = pd.cut(dataframe[features_to_keep[i]], bins=parameters[i], labels=False)


    features =  ['age', 'hours-per-week', 'capital-gain', 'capital-loss'] # [10, 10, 6, 7, 10]
    bins = parameters + [2, 2]
    bins = tuple(bins)
    # bins = (parameters[0], parameters[1], parameters[2], parameters[3], 2, 2)
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]

    return dataframe, features, sensitive_attribute, output, bins

def generate_dataset_folktables(n_samples=10000, parameters=[], seed=0):
    # To use this function, you need to define the parameters as the number of bins for each feature:
    # features order: ['AGEP', 'SCHL', 'OCCP_encoded', 'MAR_encoded', 'RELP_encoded', 'WKHP'] # MAR_encoded no more that 5 bins
    # Thus, the parameters should be a tuple with 6 elements, each element is the number of bins for each feature

    # example usage:
    # parameters = [10, 10, 10, 5, 10, 10]
    # parameters = [2, 3, 3, 2, 3, 2]
    # df, features, sensitive_attribute, output, bins = generate_dataset_folktables(n_samples=10000, parameters=parameters, seed=0)

    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')

    acs_data = data_source.get_data(states=state_list, download=True)

    # ACSIncome.features = [ 'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']

    general_features = list(set(ACSIncome.features))

    general_target = ACSIncome.target

    def general_filter(data):
        return acs.adult_filter(data)

    sensitive_attribute = 'SEX'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=general_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()

    ## encoding the categorical features using target encoding
    # OCCP
    dataframe['OCCP'] = dataframe['OCCP'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['OCCP_encoded'] = target_encoder.fit_transform(dataframe['OCCP'], dataframe[general_target])
    occ_index = dataframe.columns.get_loc('OCCP')
    dataframe.drop('OCCP', axis=1, inplace=True)
    dataframe.insert(occ_index, 'OCCP_encoded', dataframe.pop('OCCP_encoded'))
    # MAR
    dataframe['MAR'] = dataframe['MAR'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['MAR_encoded'] = target_encoder.fit_transform(dataframe['MAR'], dataframe[general_target])
    mar_index = dataframe.columns.get_loc('MAR')
    dataframe.drop('MAR', axis=1, inplace=True)
    dataframe.insert(mar_index, 'MAR_encoded', dataframe.pop('MAR_encoded'))
    # RELP
    dataframe['RELP'] = dataframe['RELP'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['RELP_encoded'] = target_encoder.fit_transform(dataframe['RELP'], dataframe[general_target])
    relp_index = dataframe.columns.get_loc('RELP')
    dataframe.drop('RELP', axis=1, inplace=True)
    dataframe.insert(relp_index, 'RELP_encoded', dataframe.pop('RELP_encoded'))

    dataframe[general_target] = dataframe[general_target].apply(lambda x: True if x > 60000 else False)

    # Drop the columns that are not needed: COW, POBP, RAC1P
    dataframe.drop('COW', axis=1, inplace=True)
    dataframe.drop('POBP', axis=1, inplace=True)
    dataframe.drop('RAC1P', axis=1, inplace=True)
    sensitive_attribute = "SEX"
    Target_label = general_target

    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 2 else 0)
    # apply bining to AGE, SCHL, OCCP_encoded, MAR_encoded, RELP_encoded, WKHP
    dataframe['AGEP'] = pd.cut(dataframe['AGEP'], bins=parameters[0], labels=False)
    dataframe['SCHL'] = pd.cut(dataframe['SCHL'], bins=parameters[1], labels=False)
    dataframe['OCCP_encoded'] = pd.cut(dataframe['OCCP_encoded'], bins=parameters[2], labels=False)
    dataframe['MAR_encoded'] = pd.cut(dataframe['MAR_encoded'], bins=parameters[3], labels=False)
    dataframe['RELP_encoded'] = pd.cut(dataframe['RELP_encoded'], bins=parameters[4], labels=False)
    dataframe['WKHP'] = pd.cut(dataframe['WKHP'], bins=parameters[5], labels=False)

    features = ['AGEP', 'SCHL', 'OCCP_encoded', 'MAR_encoded', 'RELP_encoded', 'WKHP']
    bins = (parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5],2,2)
    output = [Target_label]
    sensitive_attribute = [sensitive_attribute]

    return dataframe, features, sensitive_attribute, output, bins

def generate_dataset_folktables_selective(n_samples=10000, parameters=[], seed=0):
    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)

    # ACSIncome.features = [ 'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']

    general_features = list(set(ACSIncome.features))

    general_target = ACSIncome.target

    def general_filter(data):
        return acs.adult_filter(data)

    sensitive_attribute = 'SEX'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=general_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()

    dataframe[general_target] = dataframe[general_target].apply(lambda x: True if x > 25000 else False)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 2 else 0)

    # ## encoding the categorical features using target encoding
    # # SCHL
    dataframe['SCHL'] = dataframe['SCHL'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['SCHL_encoded'] = target_encoder.fit_transform(dataframe['SCHL'], dataframe[general_target])
    schl_index = dataframe.columns.get_loc('SCHL')
    dataframe.drop('SCHL', axis=1, inplace=True)
    dataframe.insert(schl_index, 'SCHL_encoded', dataframe.pop('SCHL_encoded'))
    # rename the column
    dataframe.rename(columns={'SCHL_encoded': 'SCHL'}, inplace=True)
    # RAC1P
    dataframe['RAC1P'] = dataframe['RAC1P'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['RAC1P_encoded'] = target_encoder.fit_transform(dataframe['RAC1P'], dataframe[general_target])
    rac1p_index = dataframe.columns.get_loc('RAC1P')
    dataframe.drop('RAC1P', axis=1, inplace=True)
    dataframe.insert(rac1p_index, 'RAC1P_encoded', dataframe.pop('RAC1P_encoded'))
    # rename the column
    dataframe.rename(columns={'RAC1P_encoded': 'RAC1P'}, inplace=True)
    # RELP
    dataframe['RELP'] = dataframe['RELP'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['RELP_encoded'] = target_encoder.fit_transform(dataframe['RELP'], dataframe[general_target])
    relp_index = dataframe.columns.get_loc('RELP')
    dataframe.drop('RELP', axis=1, inplace=True)
    dataframe.insert(relp_index, 'RELP_encoded', dataframe.pop('RELP_encoded'))
    # rename the column
    dataframe.rename(columns={'RELP_encoded': 'RELP'}, inplace=True)
    # POBP
    dataframe['POBP'] = dataframe['POBP'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['POBP_encoded'] = target_encoder.fit_transform(dataframe['POBP'], dataframe[general_target])
    pobp_index = dataframe.columns.get_loc('POBP')
    dataframe.drop('POBP', axis=1, inplace=True)
    dataframe.insert(pobp_index, 'POBP_encoded', dataframe.pop('POBP_encoded'))
    # rename the column
    dataframe.rename(columns={'POBP_encoded': 'POBP'}, inplace=True)

    features_to_use = ['SCHL', 'RAC1P', 'RELP', 'POBP', 'WKHP']
    for feature in general_features:
        if feature not in features_to_use and feature != general_target and feature != sensitive_attribute:
            dataframe.drop(feature, axis=1, inplace=True)

    # # apply bining to AGE, SCHL, OCCP_encoded, MAR_encoded, RELP_encoded, WKHP
    # dataframe['AGEP'] = pd.cut(dataframe['AGEP'], bins=10, labels=False)
    # dataframe['SCHL'] = pd.cut(dataframe['SCHL'], bins=10, labels=False)
    # dataframe['OCCP_encoded'] = pd.cut(dataframe['OCCP_encoded'], bins=10, labels=False)
    # dataframe['MAR_encoded'] = pd.cut(dataframe['MAR_encoded'], bins=5, labels=False)
    # dataframe['RELP_encoded'] = pd.cut(dataframe['RELP_encoded'], bins=10, labels=False)
    # dataframe['WKHP'] = pd.cut(dataframe['WKHP'], bins=10, labels=False)

    # apply bining to ['SCHL', 'RAC1P', 'RELP', 'POBP', 'WKHP']
    dataframe['SCHL'] = pd.cut(dataframe['SCHL'], bins=parameters[0], labels=False)
    dataframe['RAC1P'] = pd.cut(dataframe['RAC1P'], bins=parameters[1], labels=False)
    dataframe['RELP'] = pd.cut(dataframe['RELP'], bins=parameters[2], labels=False)
    dataframe['POBP'] = pd.cut(dataframe['POBP'], bins=parameters[3], labels=False)
    dataframe['WKHP'] = pd.cut(dataframe['WKHP'], bins=parameters[4], labels=False)

    features =['SCHL', 'RAC1P', 'RELP', 'POBP', 'WKHP']
    bins = (parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], 2, 2)
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]


    return dataframe, features, sensitive_attribute, output, bins

def CensusIncomeKDD(n_samples=10000, parameters=[], seed=0):
    file_path = 'datasets/dataset.arff'
    with open(file_path, 'r') as f:
        dataset = arff.load(f)

    # Extract data and convert to a pandas DataFrame
    df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])
    data = df.copy()
    # Step 1: Drop columns with more than 20% null values
    threshold = 0.2 * len(data)
    data = data.dropna(axis=1, thresh=len(data) - threshold)
    data = data.dropna()
    # Step 2: Drop rows with null values
    all_features = list(data.columns)
    target_feature = 'income_50k'
    sensitive_feature = 'sex'
    all_features.remove(target_feature)
    all_features.remove(sensitive_feature)

    categorical_features = [
        column for column in all_features if data[column].dtype == 'object'
    ]
    numerical_features = [
        column for column in all_features if data[column].dtype in ['int64', 'float64']
    ]
    for column in data.select_dtypes(['object']).columns:
        data[column] = data[column].str.strip()

    if target_feature in categorical_features:
        categorical_features.remove(target_feature)
    if target_feature in numerical_features:
        numerical_features.remove(target_feature)
    if sensitive_feature in categorical_features:
        categorical_features.remove(sensitive_feature)
    if sensitive_feature in numerical_features:
        numerical_features.remove(sensitive_feature)

    data[sensitive_feature] = data[sensitive_feature].map({'Female': 0, 'Male': 1})
    data[target_feature] = data[target_feature].map({'- 50000.': 0, '50000+.': 1})

    for feature in categorical_features:
        data[feature] = data[feature].astype('category')
        target_encoder = TargetEncoder()
        data[feature + '_encoded'] = target_encoder.fit_transform(data[feature], data[target_feature])
        occ_index = data.columns.get_loc(feature)
        data.drop(feature, axis=1, inplace=True)
        data.insert(occ_index, feature + '_encoded', data.pop(feature + '_encoded'))
        data.rename(columns={feature + '_encoded': feature}, inplace=True)

    # plot histogram for  capital_gains, capital_losses, num_emp
    # data['capital_gains'].hist(bins = 20)
    # plt.show()
    # data['capital_losses'].hist(bins= 20)
    # plt.show()
    # data['num_emp'].hist(bins = 20)
    # plt.show()

    for feature in all_features:
        data[feature] = pd.cut(data[feature], min(data[feature].nunique(), 5), labels=False)
        data[feature] = pd.cut(data[feature], data[feature].nunique(), labels=False)
    # print(categorical_features)
    all_features = ['education', 'marital_stat', 'race', 'capital_gains', 'capital_losses', 'num_emp']
    for feature in data.columns:
        if feature not in all_features and feature != target_feature and feature != sensitive_feature:
            data.drop(feature, axis=1, inplace=True)

    bins = tuple([data[feature].nunique() for feature in all_features] + [2, 2])
    output = [target_feature]
    sensitive_attribute = [sensitive_feature]

    return data, all_features, sensitive_attribute, output, bins

def CensusIncomeKDD_updated22(n_samples=10000, parameters=[], seed=0, quantization_type= 0):
    file_path = 'datasets/dataset.arff'
    with open(file_path, 'r') as f:
        dataset = arff.load(f)

    # Extract data and convert to a pandas DataFrame
    df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])
    data = df.copy()
    # Step 1: Drop columns with more than 20% null values
    threshold = 0.2 * len(data)
    data = data.dropna(axis=1, thresh=len(data) - threshold)
    data = data.dropna()
    # Step 2: Drop rows with null values
    all_features = list(data.columns)
    target_feature = 'income_50k'
    sensitive_feature = 'sex'
    all_features.remove(target_feature)
    all_features.remove(sensitive_feature)

    categorical_features = [
        column for column in all_features if data[column].dtype == 'object'
    ]
    numerical_features = [
        column for column in all_features if data[column].dtype in ['int64', 'float64']
    ]
    for column in data.select_dtypes(['object']).columns:
        data[column] = data[column].str.strip()

    if target_feature in categorical_features:
        categorical_features.remove(target_feature)
    if target_feature in numerical_features:
        numerical_features.remove(target_feature)
    if sensitive_feature in categorical_features:
        categorical_features.remove(sensitive_feature)
    if sensitive_feature in numerical_features:
        numerical_features.remove(sensitive_feature)

    data[sensitive_feature] = data[sensitive_feature].map({'Female': 0, 'Male': 1})
    data[target_feature] = data[target_feature].map({'- 50000.': 0, '50000+.': 1})

    for feature in categorical_features:
        data[feature] = data[feature].astype('category')
        target_encoder = TargetEncoder()
        data[feature + '_encoded'] = target_encoder.fit_transform(data[feature], data[target_feature])
        occ_index = data.columns.get_loc(feature)
        data.drop(feature, axis=1, inplace=True)
        data.insert(occ_index, feature + '_encoded', data.pop(feature + '_encoded'))
        data.rename(columns={feature + '_encoded': feature}, inplace=True)

    # plot histogram for  capital_gains, capital_losses, num_emp
    # data['capital_gains'].hist(bins = 20)
    # plt.show()
    # data['capital_losses'].hist(bins= 20)
    # plt.show()
    # data['num_emp'].hist(bins = 20)
    # plt.show()
    continous_features = ['capital_gains', 'capital_losses', 'num_emp']

    for feature in all_features:
        if feature in continous_features:
            continue
        data[feature] = pd.cut(data[feature], min(data[feature].nunique(), 5), labels=False)
        data[feature] = pd.cut(data[feature], data[feature].nunique(), labels=False)

    if quantization_type == 0:
        for feature in continous_features:
            data[feature] = pd.cut(data[feature], 2, labels=False)
    else: # equal fequency binning
        for feature in continous_features:
            data[feature] = pd.qcut(data[feature], 2, labels=False, duplicates='drop')



    # print(categorical_features)
    all_features = ['education', 'marital_stat', 'race', 'capital_gains', 'capital_losses', 'num_emp']
    for feature in data.columns:
        if feature not in all_features and feature != target_feature and feature != sensitive_feature:
            data.drop(feature, axis=1, inplace=True)

    # bins = tuple([data[feature].nunique() for feature in all_features] + [2, 2])
    bins = tuple([data[feature].nunique() for feature in all_features] + [2, 2])
    output = [target_feature]
    sensitive_attribute = [sensitive_feature]

    return data, all_features, sensitive_attribute, output, bins

def CensusIncomeKDD_updated(n_samples=10000, parameters=[], seed=0):
    file_path = 'datasets/dataset.arff'
    with open(file_path, 'r') as f:
        dataset = arff.load(f)

    # Extract data and convert to a pandas DataFrame
    df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])


    # print(df.head())
    data = df.copy()
    # Step 1: Drop columns with more than 20% null values
    threshold = 0.2 * len(data)
    data = data.dropna(axis=1, thresh=len(data) - threshold)
    data = data.dropna()
    # Step 2: Drop rows with null values
    all_features = list(data.columns)
    target_feature = 'income_50k'
    sensitive_feature = 'sex'
    all_features.remove(target_feature)
    all_features.remove(sensitive_feature)

    categorical_features = [
        column for column in all_features if data[column].dtype == 'object'
    ]
    numerical_features = [
        column for column in all_features if data[column].dtype in ['int64', 'float64']
    ]
    for column in data.select_dtypes(['object']).columns:
        data[column] = data[column].str.strip()

    if target_feature in categorical_features:
        categorical_features.remove(target_feature)
    if target_feature in numerical_features:
        numerical_features.remove(target_feature)
    if sensitive_feature in categorical_features:
        categorical_features.remove(sensitive_feature)
    if sensitive_feature in numerical_features:
        numerical_features.remove(sensitive_feature)

    data[sensitive_feature] = data[sensitive_feature].map({'Female': 0, 'Male': 1})
    data[target_feature] = data[target_feature].map({'- 50000.': 0, '50000+.': 1})

    # for feature in categorical_features:
    #     data[feature] = data[feature].astype('category')
    #     target_encoder = TargetEncoder()
    #     data[feature + '_encoded'] = target_encoder.fit_transform(data[feature], data[target_feature])
    #     occ_index = data.columns.get_loc(feature)
    #     data.drop(feature, axis=1, inplace=True)
    #     data.insert(occ_index, feature + '_encoded', data.pop(feature + '_encoded'))
    #     data.rename(columns={feature + '_encoded': feature}, inplace=True)

    for feature in categorical_features:
        data[feature] = data[feature].astype('category')
        target_encoder = LabelEncoder()
        data[feature + '_encoded'] = target_encoder.fit_transform(data[feature])
        occ_index = data.columns.get_loc(feature)
        data.drop(feature, axis=1, inplace=True)
        data.insert(occ_index, feature + '_encoded', data.pop(feature + '_encoded'))
        data.rename(columns={feature + '_encoded': feature}, inplace=True)


    # plot histogram for  capital_gains, capital_losses, num_emp
    # data['capital_gains'].hist(bins = 20)
    # plt.show()
    # data['capital_losses'].hist(bins= 20)
    # plt.show()
    # data['num_emp'].hist(bins = 20)
    # plt.show()
    continous_features = ['capital_gains', 'capital_losses', 'num_emp']

    # for feature in all_features:
    #     if feature in continous_features:
    #         continue
    #     print(feature)
    #     print(data[feature].nunique())
        # transform categorical features to numerical

        # data[feature] = pd.cut(data[feature], min(data[feature].nunique(), 20), labels=False)
        # data[feature] = pd.cut(data[feature], data[feature].nunique(), labels=False)

    data['num_emp'] = pd.qcut(data['num_emp'], 10, labels=False, duplicates='drop')
    data['capital_gains'] = pd.cut(data['capital_gains'], 2, labels=False)
    data['capital_losses'] = pd.cut(data['capital_losses'], 2, labels=False)


    # print(categorical_features)
    all_features = ['education', 'marital_stat', 'race', 'capital_gains', 'capital_losses', 'num_emp']
    for feature in data.columns:
        if feature not in all_features and feature != target_feature and feature != sensitive_feature:
            data.drop(feature, axis=1, inplace=True)

    # bins = tuple([data[feature].nunique() for feature in all_features] + [2, 2])
    bins = tuple([data[feature].nunique() for feature in all_features] + [2, 2])
    output = [target_feature]
    sensitive_attribute = [sensitive_feature]

    return data, all_features, sensitive_attribute, output, bins, ['education', 'marital_stat', 'race']


def generate_dataset_folktables_PublicCoverage_selective_new(n_samples=10000, parameters=[], seed=0):
    all = True
    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)

    # ACSIncome.features = [ 'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']

    general_features = list(set(ACSPublicCoverage.features))

    general_target = ACSPublicCoverage.target

    def general_filter(data):
        return acs.public_coverage_filter(data)

    sensitive_attribute = 'RAC1P'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=acs.travel_time_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()
    # dataframe = dataframe[dataframe[sensitive_attribute].isin([1, 2])]
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x == 1 else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 1 else 0)
    # positive_rate for A = 1 and A = 0
    positive_rate_A1 = dataframe[(dataframe[sensitive_attribute] == 1) & (dataframe[general_target] == 1)].shape[0] / \
                       dataframe[dataframe[sensitive_attribute] == 1].shape[0]
    positive_rate_A0 = dataframe[(dataframe[sensitive_attribute] == 0) & (dataframe[general_target] == 1)].shape[0] / \
                       dataframe[dataframe[sensitive_attribute] == 0].shape[0]

    # print(f"Positive rate for A=1: {positive_rate_A1}")
    # print(f"Positive rate for A=0: {positive_rate_A0}")
    features_to_use = [feature for feature in dataframe.columns if
                       feature != general_target and feature != sensitive_attribute]
    if all:
        # features_to_use = [MAR, PINCP, ESR, ST, NATIVITY, DREM]
        features_to_use = ['NATIVITY', 'CIT', 'ESR', 'PINCP', 'SCHL', 'MIG', 'AGEP', 'ST', 'MIL', 'ANC', 'DEYE', 'DIS',
                           'FER', 'DEAR', 'DREM', 'SEX', 'MAR', 'ESP']
        categorical_features = ['NATIVITY', 'CIT', 'ESR', 'SCHL', 'MIG', 'ST', 'MIL', 'ANC', 'DEYE', 'DIS', 'FER',
                                'DEAR', 'DREM', 'SEX', 'MAR', 'ESP']

        for feature in dataframe.columns:
            if feature not in features_to_use and feature != general_target and feature != sensitive_attribute:
                dataframe.drop(feature, axis=1, inplace=True)

        # print(dataframe.columns)

        number_of_unique_values = {}
        for feature in dataframe.columns:
            if feature != general_target and feature != sensitive_attribute:
                if feature in categorical_features:
                    number_of_unique_values[feature] = dataframe[feature].nunique()
                else:
                    number_of_unique_values[feature] = 100

        bins = {}
        for feature in number_of_unique_values.keys():
            bins[feature] = min(5, number_of_unique_values[feature])

        for feature in categorical_features:
            dataframe[feature] = dataframe[feature].astype('category')
            target_encoder = TargetEncoder()
            dataframe[feature + '_encoded'] = target_encoder.fit_transform(dataframe[feature],
                                                                           dataframe[general_target])
            occ_index = dataframe.columns.get_loc(feature)
            dataframe.drop(feature, axis=1, inplace=True)
            dataframe.insert(occ_index, feature + '_encoded', dataframe.pop(feature + '_encoded'))
            dataframe.rename(columns={feature + '_encoded': feature}, inplace=True)
            # dataframe.rename(columns={'SCHL_encoded': 'SCHL'}, inplace=True)

        # print(dataframe.head())
        # apply binning
        for feature in dataframe.columns:
            if feature != general_target and feature != sensitive_attribute:
                dataframe[feature] = pd.cut(dataframe[feature], bins[feature], labels=False)

    # #
    for feature in features_to_use:
        bins[feature] = dataframe[feature].nunique()

    bins = tuple(list(bins.values())+[2,2])

    features = features_to_use
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]
    return dataframe, features, sensitive_attribute, output, bins


def generate_dataset_folktables_PublicCoverage_selective(n_samples=10000, parameters=[], seed=0):
    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)

    # ACSIncome.features = [ 'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']

    general_features = list(set(ACSPublicCoverage.features))

    general_target = ACSPublicCoverage.target

    def general_filter(data):
        return acs.public_coverage_filter(data)

    sensitive_attribute = 'DIS'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=general_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x == 1 else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 2 else 0)

    # features_to_use = [MAR, PINCP, ESR, ST, NATIVITY, DREM]
    features_to_use = ['MAR', 'PINCP', 'ESR', 'ST', 'DREM']
    categorical_features = ['MAR', 'ESR', 'ST', 'DREM']

    for feature in dataframe.columns:
        if feature not in features_to_use and feature != general_target and feature != sensitive_attribute:
            dataframe.drop(feature, axis=1, inplace=True)

    # print(dataframe.columns)

    number_of_unique_values = {}
    for feature in dataframe.columns:
        if feature != general_target and feature != sensitive_attribute:
            if feature in categorical_features:
                number_of_unique_values[feature] = dataframe[feature].nunique()
            else:
                number_of_unique_values[feature] = 100

    # print(number_of_unique_values)
    bins = {}
    for feature in number_of_unique_values.keys():
        bins[feature] = min(5, number_of_unique_values[feature])

    for feature in categorical_features:
        if number_of_unique_values[feature] > bins[feature]:
            dataframe[feature] = dataframe[feature].astype('category')
            target_encoder = TargetEncoder()
            dataframe[feature + '_encoded'] = target_encoder.fit_transform(dataframe[feature],
                                                                           dataframe[general_target])
            occ_index = dataframe.columns.get_loc(feature)
            dataframe.drop(feature, axis=1, inplace=True)
            dataframe.insert(occ_index, feature + '_encoded', dataframe.pop(feature + '_encoded'))
            dataframe.rename(columns={feature + '_encoded': feature}, inplace=True)
            # dataframe.rename(columns={'SCHL_encoded': 'SCHL'}, inplace=True)

    # apply binning
    for feature in dataframe.columns:
        if feature != general_target and feature != sensitive_attribute:
            dataframe[feature] = pd.cut(dataframe[feature], bins[feature], labels=False)

    bins = tuple([bins[feature] for feature in dataframe.columns if feature != general_target and feature != sensitive_attribute]+[2, 2])
    features = [feature for feature in dataframe.columns if feature != general_target and feature != sensitive_attribute]
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]
    # print(features)
    # print(sensitive_attribute)
    # print(output)
    # print(bins)
    # print(dataframe.head())
    return dataframe, features, sensitive_attribute, output, bins

def generate_dataset_folktables_PublicCoverage_selective_2(n_samples=10000, parameters=[], seed=0):
    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)

    # ACSIncome.features = [ 'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']

    general_features = list(set(ACSPublicCoverage.features))

    general_target = ACSPublicCoverage.target

    def general_filter(data):
        return acs.public_coverage_filter(data)

    sensitive_attribute = 'RAC1P'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=general_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x == 1 else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 1 else 0)

    # # plot histogram of the AGEP
    # dataframe['AGEP'].hist(bins=20)
    # plt.show()

    # features_to_use = [MAR, PINCP, ESR, ST, NATIVITY, DREM]
    features_to_use = ['AGEP', 'ESP', 'SEX', 'MIG', 'FER']
    categorical_features = ['ESP', 'SEX', 'MIG', 'FER']

    for feature in dataframe.columns:
        if feature not in features_to_use and feature != general_target and feature != sensitive_attribute:
            dataframe.drop(feature, axis=1, inplace=True)

    # print(dataframe.columns)

    number_of_unique_values = {}
    for feature in dataframe.columns:
        if feature != general_target and feature != sensitive_attribute:
            if feature in categorical_features:
                number_of_unique_values[feature] = dataframe[feature].nunique()
            else:
                number_of_unique_values[feature] = 100

    # print(number_of_unique_values)
    bins = {}
    for feature in number_of_unique_values.keys():
        bins[feature] = min(5, number_of_unique_values[feature])

    for feature in categorical_features:
        if number_of_unique_values[feature] > bins[feature]:
            dataframe[feature] = dataframe[feature].astype('category')
            target_encoder = TargetEncoder()
            dataframe[feature + '_encoded'] = target_encoder.fit_transform(dataframe[feature],
                                                                           dataframe[general_target])
            occ_index = dataframe.columns.get_loc(feature)
            dataframe.drop(feature, axis=1, inplace=True)
            dataframe.insert(occ_index, feature + '_encoded', dataframe.pop(feature + '_encoded'))
            dataframe.rename(columns={feature + '_encoded': feature}, inplace=True)
            # dataframe.rename(columns={'SCHL_encoded': 'SCHL'}, inplace=True)

    # apply binning
    for feature in dataframe.columns:
        if feature != general_target and feature != sensitive_attribute:
            dataframe[feature] = pd.cut(dataframe[feature], bins[feature], labels=False)

    bins = tuple([bins[feature] for feature in features_to_use]+[2, 2])
    features = features_to_use
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]
    # print(features)
    # print(sensitive_attribute)
    # print(output)
    # print(bins)
    # print(dataframe.head())
    return dataframe, features, sensitive_attribute, output, bins


def generate_dataset_folktables_PublicCoverage_selective_2_updated(n_samples=10000, parameters=[], seed=0):
    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)

    # ACSIncome.features = [ 'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']

    general_features = list(set(ACSPublicCoverage.features))

    general_target = ACSPublicCoverage.target

    def general_filter(data):
        return acs.public_coverage_filter(data)

    sensitive_attribute = 'RAC1P'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=general_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x == 1 else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 1 else 0)

    # # plot histogram of the AGEP
    # dataframe['AGEP'].hist(bins=20)
    # plt.show()

    # features_to_use = [MAR, PINCP, ESR, ST, NATIVITY, DREM]
    features_to_use = ['AGEP', 'ESP', 'SEX', 'MIG', 'FER']
    categorical_features = ['ESP', 'SEX', 'MIG', 'FER']

    for feature in dataframe.columns:
        if feature not in features_to_use and feature != general_target and feature != sensitive_attribute:
            dataframe.drop(feature, axis=1, inplace=True)

    # print(dataframe.columns)

    number_of_unique_values = {}
    for feature in dataframe.columns:
        if feature != general_target and feature != sensitive_attribute:
            if feature in categorical_features:
                number_of_unique_values[feature] = dataframe[feature].nunique()
            else:
                number_of_unique_values[feature] = 100

    # print(number_of_unique_values)
    bins = {}
    for feature in number_of_unique_values.keys():
        bins[feature] = min(10, number_of_unique_values[feature])

    # for feature in categorical_features:
    #     if number_of_unique_values[feature] > bins[feature]:
    #         dataframe[feature] = dataframe[feature].astype('category')
    #         target_encoder = TargetEncoder()
    #         dataframe[feature + '_encoded'] = target_encoder.fit_transform(dataframe[feature],
    #                                                                        dataframe[general_target])
    #         occ_index = dataframe.columns.get_loc(feature)
    #         dataframe.drop(feature, axis=1, inplace=True)
    #         dataframe.insert(occ_index, feature + '_encoded', dataframe.pop(feature + '_encoded'))
    #         dataframe.rename(columns={feature + '_encoded': feature}, inplace=True)
    #         # dataframe.rename(columns={'SCHL_encoded': 'SCHL'}, inplace=True)

    # apply binning
    continues_features = ['AGEP']
    for feature in dataframe.columns:
        if feature in continues_features:
            dataframe[feature] = pd.qcut(dataframe[feature], q=bins[feature], labels=False)

    bins = tuple([dataframe[feature].nunique() for feature in features_to_use]+[2, 2])
    features = features_to_use
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]
    # print(features)
    # print(sensitive_attribute)
    # print(output)
    # print(bins)
    # print(dataframe.head())
    return dataframe, features, sensitive_attribute, output, bins, categorical_features

def generate_dataset_folktables_RAC1P(n_samples=10000, parameters=[], seed=0):
    # To use this function, you need to define the parameters as the number of bins for each feature:
    # features order: ['AGEP', 'SCHL', 'OCCP_encoded', 'MAR_encoded', 'RELP_encoded', 'WKHP'] # MAR_encoded no more that 5 bins
    # Thus, the parameters should be a tuple with 6 elements, each element is the number of bins for each feature

    # example usage:
    # parameters = [10, 10, 10, 5, 10, 10]
    # parameters = [2, 3, 3, 2, 3, 2]
    # df, features, sensitive_attribute, output, bins = generate_dataset_folktables(n_samples=10000, parameters=parameters, seed=0)

    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)

    # ACSIncome.features = [ 'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']

    general_features = list(set(ACSIncome.features))

    general_target = ACSIncome.target

    def general_filter(data):
        return acs.adult_filter(data)

    sensitive_attribute = 'RAC1P'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=general_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()

    ## encoding the categorical features using target encoding
    # OCCP
    dataframe['OCCP'] = dataframe['OCCP'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['OCCP_encoded'] = target_encoder.fit_transform(dataframe['OCCP'], dataframe[general_target])
    occ_index = dataframe.columns.get_loc('OCCP')
    dataframe.drop('OCCP', axis=1, inplace=True)
    dataframe.insert(occ_index, 'OCCP_encoded', dataframe.pop('OCCP_encoded'))
    # MAR
    dataframe['MAR'] = dataframe['MAR'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['MAR_encoded'] = target_encoder.fit_transform(dataframe['MAR'], dataframe[general_target])
    mar_index = dataframe.columns.get_loc('MAR')
    dataframe.drop('MAR', axis=1, inplace=True)
    dataframe.insert(mar_index, 'MAR_encoded', dataframe.pop('MAR_encoded'))
    # RELP
    dataframe['RELP'] = dataframe['RELP'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['RELP_encoded'] = target_encoder.fit_transform(dataframe['RELP'], dataframe[general_target])
    relp_index = dataframe.columns.get_loc('RELP')
    dataframe.drop('RELP', axis=1, inplace=True)
    dataframe.insert(relp_index, 'RELP_encoded', dataframe.pop('RELP_encoded'))

    dataframe[general_target] = dataframe[general_target].apply(lambda x: True if x > 50000 else False)

    # Drop the columns that are not needed: COW, POBP, RAC1P
    dataframe.drop('COW', axis=1, inplace=True)
    dataframe.drop('POBP', axis=1, inplace=True)
    dataframe.drop('SEX', axis=1, inplace=True)
    sensitive_attribute = "RAC1P"
    Target_label = general_target

    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 1 else 0)
    # apply bining to AGE, SCHL, OCCP_encoded, MAR_encoded, RELP_encoded, WKHP
    dataframe['AGEP'] = pd.cut(dataframe['AGEP'], bins=parameters[0], labels=False)
    dataframe['SCHL'] = pd.cut(dataframe['SCHL'], bins=parameters[1], labels=False)
    dataframe['OCCP_encoded'] = pd.cut(dataframe['OCCP_encoded'], bins=parameters[2], labels=False)
    dataframe['MAR_encoded'] = pd.cut(dataframe['MAR_encoded'], bins=parameters[3], labels=False)
    dataframe['RELP_encoded'] = pd.cut(dataframe['RELP_encoded'], bins=parameters[4], labels=False)
    dataframe['WKHP'] = pd.cut(dataframe['WKHP'], bins=parameters[5], labels=False)

    features = ['AGEP', 'SCHL', 'OCCP_encoded', 'MAR_encoded', 'RELP_encoded', 'WKHP']
    bins = (parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5],2,2)
    output = [Target_label]
    sensitive_attribute = [sensitive_attribute]

    return dataframe, features, sensitive_attribute, output, bins

def generate_dataset_folktables_RAC1P_WB(n_samples=10000, parameters=[], seed=0):
    # To use this function, you need to define the parameters as the number of bins for each feature:
    # features order: ['AGEP', 'SCHL', 'OCCP_encoded', 'MAR_encoded', 'RELP_encoded', 'WKHP'] # MAR_encoded no more that 5 bins
    # Thus, the parameters should be a tuple with 6 elements, each element is the number of bins for each feature

    # example usage:
    # parameters = [10, 10, 10, 5, 10, 10]
    # parameters = [2, 3, 3, 2, 3, 2]
    # df, features, sensitive_attribute, output, bins = generate_dataset_folktables(n_samples=10000, parameters=parameters, seed=0)

    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)

    # ACSIncome.features = [ 'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']

    general_features = list(set(ACSIncome.features))

    general_target = ACSIncome.target

    def general_filter(data):
        return acs.adult_filter(data)

    sensitive_attribute = 'RAC1P'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=general_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()

    dataframe = dataframe[dataframe[sensitive_attribute].isin([1, 2])]

    ## encoding the categorical features using target encoding
    # OCCP
    dataframe['OCCP'] = dataframe['OCCP'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['OCCP_encoded'] = target_encoder.fit_transform(dataframe['OCCP'], dataframe[general_target])
    occ_index = dataframe.columns.get_loc('OCCP')
    dataframe.drop('OCCP', axis=1, inplace=True)
    dataframe.insert(occ_index, 'OCCP_encoded', dataframe.pop('OCCP_encoded'))
    # MAR
    dataframe['MAR'] = dataframe['MAR'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['MAR_encoded'] = target_encoder.fit_transform(dataframe['MAR'], dataframe[general_target])
    mar_index = dataframe.columns.get_loc('MAR')
    dataframe.drop('MAR', axis=1, inplace=True)
    dataframe.insert(mar_index, 'MAR_encoded', dataframe.pop('MAR_encoded'))
    # RELP
    dataframe['RELP'] = dataframe['RELP'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['RELP_encoded'] = target_encoder.fit_transform(dataframe['RELP'], dataframe[general_target])
    relp_index = dataframe.columns.get_loc('RELP')
    dataframe.drop('RELP', axis=1, inplace=True)
    dataframe.insert(relp_index, 'RELP_encoded', dataframe.pop('RELP_encoded'))

    dataframe[general_target] = dataframe[general_target].apply(lambda x: True if x > 50000 else False)

    # Drop the columns that are not needed: COW, POBP, RAC1P
    dataframe.drop('COW', axis=1, inplace=True)
    dataframe.drop('POBP', axis=1, inplace=True)
    dataframe.drop('SEX', axis=1, inplace=True)
    sensitive_attribute = "RAC1P"
    Target_label = general_target

    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 1 else 0)
    # apply bining to AGE, SCHL, OCCP_encoded, MAR_encoded, RELP_encoded, WKHP
    dataframe['AGEP'] = pd.cut(dataframe['AGEP'], bins=parameters[0], labels=False)
    dataframe['SCHL'] = pd.cut(dataframe['SCHL'], bins=parameters[1], labels=False)
    dataframe['OCCP_encoded'] = pd.cut(dataframe['OCCP_encoded'], bins=parameters[2], labels=False)
    dataframe['MAR_encoded'] = pd.cut(dataframe['MAR_encoded'], bins=parameters[3], labels=False)
    dataframe['RELP_encoded'] = pd.cut(dataframe['RELP_encoded'], bins=parameters[4], labels=False)
    dataframe['WKHP'] = pd.cut(dataframe['WKHP'], bins=parameters[5], labels=False)

    features = ['AGEP', 'SCHL', 'OCCP_encoded', 'MAR_encoded', 'RELP_encoded', 'WKHP']
    bins = (parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5],2,2)
    output = [Target_label]
    sensitive_attribute = [sensitive_attribute]

    return dataframe, features, sensitive_attribute, output, bins

def Heritage(n_samples=10000, parameters=[], seed=0):
    # Load your claims dataset
    data = pd.read_csv('datasets/Claims.txt')  # Replace with your file path
    # print(data.columns)
    data_member = pd.read_csv('datasets/Members.txt')  # Replace with your file path
    # print(data.columns)

    common_column = 'MemberID'
    # print(data_member.columns)
    # Specify the features to add from data_member
    features_to_add = ['AgeAtFirstClaim', 'Sex']  # Replace with actual feature names

    # Perform the merge operation
    data = data.merge(data_member[[common_column] + features_to_add], on=common_column, how='left')

    # Display the updated columns to confirm
    # print(data.columns)

    data = data.dropna()

    # Dropping specified columns
    columns_to_drop = ["MemberID", "ProviderID", "Vendor", "PCP"]
    data = data.drop(columns=columns_to_drop, errors="ignore")

    # Function to convert period to days
    def convert_to_days(value, mapping):
        if pd.isna(value):
            return np.nan
        return mapping.get(value, np.nan)

    # LengthOfStay conversion
    length_of_stay_mapping = {
        '1 day': 1,
        '2 days': 2,
        '3 days': 3,
        '4 days': 4,
        '5 days': 5,
        '6 days': 6,
        '1- 2 weeks': 7,
        '2- 4 weeks': 14,
        '4- 8 weeks': 30,
        '26+ weeks': 182,
    }
    data['LengthOfStay'] = data['LengthOfStay'].apply(lambda x: convert_to_days(x, length_of_stay_mapping))

    # DSFS conversion
    dsfs_mapping = {
        '0- 1 month': 15,
        '1- 2 months': 45,
        '2- 3 months': 75,
        '3- 4 months': 105,
        '4- 5 months': 135,
        '5- 6 months': 165,
        '6- 7 months': 195,
        '7- 8 months': 225,
        '8- 9 months': 255,
        '9-10 months': 285,
        '10-11 months': 315,
        '11-12 months': 345,
    }
    data['DSFS'] = data['DSFS'].apply(lambda x: convert_to_days(x, dsfs_mapping))

    # Converting categorical features to numerical indices
    categorical_columns = ['Year', 'Specialty', 'PlaceSvc', 'PrimaryConditionGroup', 'ProcedureGroup']
    for column in categorical_columns:
        if column in data.columns:
            data[column] = pd.Categorical(data[column]).codes

    # Explicitly mapping CharlsonIndex
    charlson_index_mapping = {'0': 0, '1-2': 1, '3-4': 2, '5+': 3}
    data['CharlsonIndex'] = data['CharlsonIndex'].map(charlson_index_mapping)

    def convert_pay_delay(value):
        if value == '162+':
            return 162
        try:
            return int(value)
        except ValueError:
            return np.nan  # Handle unexpected values

    # Converting PayDelay to numerical values
    if 'PayDelay' in data.columns:
        data['PayDelay'] = data['PayDelay'].apply(convert_pay_delay)

    features_total = ['Year', 'Specialty',
                      'PlaceSvc', 'PayDelay', 'LengthOfStay', 'DSFS', 'PrimaryConditionGroup',
                      'CharlsonIndex', 'ProcedureGroup', 'SupLOS']

    categorical_features = ['Year', 'Specialty', 'PlaceSvc', 'PrimaryConditionGroup', 'CharlsonIndex',
                            'ProcedureGroup']  #
    num_categories = { # ['Year', 'PayDelay', 'PlaceSvc', 'DSFS', 'Sex']
        'Year': 3,  # [0, 1, 2]
        'Specialty': 4,  # [0, 1, 2, 3]
        'PlaceSvc': 7,  # [0, 1, 2, 3, 4, 5, 6]
        'PrimaryConditionGroup': 43,  # [0, 1, ..., 42]
        'CharlsonIndex': 4,  # [0, 1, 2, 3]
        'ProcedureGroup': 15  # [0, 1, ..., 14]
    }

    age_mapping = {
        '0-9': 0,
        '10-19': 1,
        '20-29': 2,
        '30-39': 3,
        '40-49': 4,
        '50-59': 5,
        '60-69': 6,
        '70-79': 7,
        '80+': 8
    }

    # Mapping Sex to numerical indices
    sex_mapping = {
        'M': 0,
        'F': 1
    }
    data['AgeAtFirstClaim'] = data['AgeAtFirstClaim'].map(age_mapping)
    data['Sex'] = data['Sex'].map(sex_mapping)

    # for feature in data.columns: # unique values
    #     print(f'{feature}: {data[feature].unique()}')
    dataframe = data.copy()

    # sensitive_attribute = 'Sex'
    sensitive_attribute = 'AgeAtFirstClaim'
    # threshold age at 80
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x >= 8 else 0)
    general_target = 'CharlsonIndex'
    # convert CharlsonIndex to binary to detect zero
    dataframe = dataframe.replace({'CharlsonIndex': {0: 0, 1: 1, 2: 1, 3: 1}})
    features_to_use = [feature for feature in data.columns if feature not in [sensitive_attribute, general_target]]

    features_to_use = ['Sex', 'Year', 'PlaceSvc', 'PayDelay', 'DSFS']  # 2 3 6 10 12
    categoercal = ['Sex', 'Year', 'PlaceSvc']  # 2, 3, 7, 12
    bins_for_features_to_use = {'Sex': 2, 'Year': 3, 'PlaceSvc': 7, 'PayDelay': 10, 'DSFS': 12}
    # print(dataframe['DSFS'].value_counts())
    # plot histogram for 'PayDelay'
    # plt.hist(dataframe['PayDelay'], bins=10)
    # plt.show()
    for feature in dataframe.columns:
        if feature not in features_to_use and feature != general_target and feature != sensitive_attribute:
            dataframe = dataframe.drop(feature, axis=1)

    for feature in dataframe.columns:
        if feature != general_target and feature != sensitive_attribute:
            dataframe[feature] = pd.cut(dataframe[feature], bins_for_features_to_use[feature], labels=False)

    bins = tuple([bins_for_features_to_use[feature] for feature in ['Year', 'PayDelay', 'PlaceSvc', 'DSFS', 'Sex']] + [2, 2])
    features = ['Year', 'PayDelay', 'PlaceSvc', 'DSFS', 'Sex']
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]
    return dataframe, features, sensitive_attribute, output, bins


def Heritage_updated(n_samples=10000, parameters=[], seed=0):
    # Load your claims dataset
    data = pd.read_csv('datasets/Claims.txt')  # Replace with your file path
    # print(data.columns)
    data_member = pd.read_csv('datasets/Members.txt')  # Replace with your file path
    # print(data.columns)

    common_column = 'MemberID'
    # print(data_member.columns)
    # Specify the features to add from data_member
    features_to_add = ['AgeAtFirstClaim', 'Sex']  # Replace with actual feature names

    # Perform the merge operation
    data = data.merge(data_member[[common_column] + features_to_add], on=common_column, how='left')

    # Display the updated columns to confirm
    # print(data.columns)

    data = data.dropna()

    # Dropping specified columns
    columns_to_drop = ["MemberID", "ProviderID", "Vendor", "PCP"]
    data = data.drop(columns=columns_to_drop, errors="ignore")

    # Function to convert period to days
    def convert_to_days(value, mapping):
        if pd.isna(value):
            return np.nan
        return mapping.get(value, np.nan)

    # LengthOfStay conversion
    length_of_stay_mapping = {
        '1 day': 1,
        '2 days': 2,
        '3 days': 3,
        '4 days': 4,
        '5 days': 5,
        '6 days': 6,
        '1- 2 weeks': 7,
        '2- 4 weeks': 14,
        '4- 8 weeks': 30,
        '26+ weeks': 182,
    }
    data['LengthOfStay'] = data['LengthOfStay'].apply(lambda x: convert_to_days(x, length_of_stay_mapping))

    # DSFS conversion
    dsfs_mapping = {
        '0- 1 month': 15,
        '1- 2 months': 45,
        '2- 3 months': 75,
        '3- 4 months': 105,
        '4- 5 months': 135,
        '5- 6 months': 165,
        '6- 7 months': 195,
        '7- 8 months': 225,
        '8- 9 months': 255,
        '9-10 months': 285,
        '10-11 months': 315,
        '11-12 months': 345,
    }
    data['DSFS'] = data['DSFS'].apply(lambda x: convert_to_days(x, dsfs_mapping))

    # Converting categorical features to numerical indices
    categorical_columns = ['Year', 'Specialty', 'PlaceSvc', 'PrimaryConditionGroup', 'ProcedureGroup']
    for column in categorical_columns:
        if column in data.columns:
            data[column] = pd.Categorical(data[column]).codes

    # Explicitly mapping CharlsonIndex
    charlson_index_mapping = {'0': 0, '1-2': 1, '3-4': 2, '5+': 3}
    data['CharlsonIndex'] = data['CharlsonIndex'].map(charlson_index_mapping)

    def convert_pay_delay(value):
        if value == '162+':
            return 162
        try:
            return int(value)
        except ValueError:
            return np.nan  # Handle unexpected values

    # Converting PayDelay to numerical values
    if 'PayDelay' in data.columns:
        data['PayDelay'] = data['PayDelay'].apply(convert_pay_delay)

    features_total = ['Year', 'Specialty',
                      'PlaceSvc', 'PayDelay', 'LengthOfStay', 'DSFS', 'PrimaryConditionGroup',
                      'CharlsonIndex', 'ProcedureGroup', 'SupLOS']

    categorical_features = ['Year', 'Specialty', 'PlaceSvc', 'PrimaryConditionGroup', 'CharlsonIndex',
                            'ProcedureGroup']  #
    num_categories = { # ['Year', 'PayDelay', 'PlaceSvc', 'DSFS', 'Sex']
        'Year': 3,  # [0, 1, 2]
        'Specialty': 4,  # [0, 1, 2, 3]
        'PlaceSvc': 7,  # [0, 1, 2, 3, 4, 5, 6]
        'PrimaryConditionGroup': 43,  # [0, 1, ..., 42]
        'CharlsonIndex': 4,  # [0, 1, 2, 3]
        'ProcedureGroup': 15  # [0, 1, ..., 14]
    }

    age_mapping = {
        '0-9': 0,
        '10-19': 1,
        '20-29': 2,
        '30-39': 3,
        '40-49': 4,
        '50-59': 5,
        '60-69': 6,
        '70-79': 7,
        '80+': 8
    }

    # Mapping Sex to numerical indices
    sex_mapping = {
        'M': 0,
        'F': 1
    }
    data['AgeAtFirstClaim'] = data['AgeAtFirstClaim'].map(age_mapping)
    data['Sex'] = data['Sex'].map(sex_mapping)

    # for feature in data.columns: # unique values
    #     print(f'{feature}: {data[feature].unique()}')
    dataframe = data.copy()

    # sensitive_attribute = 'Sex'
    sensitive_attribute = 'AgeAtFirstClaim'
    # threshold age at 80
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x >= 8 else 0)
    general_target = 'CharlsonIndex'
    # convert CharlsonIndex to binary to detect zero
    dataframe = dataframe.replace({'CharlsonIndex': {0: 0, 1: 1, 2: 1, 3: 1}})
    features_to_use = [feature for feature in data.columns if feature not in [sensitive_attribute, general_target]]

    features_to_use = ['Sex', 'Year', 'PlaceSvc', 'PayDelay', 'DSFS']  # 2 3 6 10 12
    categoercal = ['Sex', 'Year', 'PlaceSvc']  # 2, 3, 7, 12
    bins_for_features_to_use = {'Sex': 2, 'Year': 3, 'PlaceSvc': 7, 'PayDelay': 10, 'DSFS': 12}
    # print(dataframe['DSFS'].value_counts())
    # # plot histogram for 'PayDelay'
    # plt.hist(dataframe['PayDelay'], bins=10)
    # plt.show()
    for feature in dataframe.columns:
        if feature not in features_to_use and feature != general_target and feature != sensitive_attribute:
            dataframe = dataframe.drop(feature, axis=1)

    for feature in dataframe.columns:
        if feature in categoercal:
            continue
        if feature != general_target and feature != sensitive_attribute:
            dataframe[feature] = pd.qcut(dataframe[feature], q = bins_for_features_to_use[feature], labels=False, duplicates='drop')
            bins_for_features_to_use[feature] = dataframe[feature].unique().shape[0]

    bins = tuple([bins_for_features_to_use[feature] for feature in ['Year', 'PayDelay', 'PlaceSvc', 'DSFS', 'Sex']] + [2, 2])
    features = ['Year', 'PayDelay', 'PlaceSvc', 'DSFS', 'Sex']
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]
    return dataframe, features, sensitive_attribute, output, bins, categoercal

def generate_dataset_folktables_RAC1P_WB_selective(n_samples=10000, parameters=[], seed=0):
    all = True
    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)

    # ACSIncome.features = [ 'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']

    general_features = list(set(ACSIncome.features))

    general_target = ACSIncome.target

    def general_filter(data):
        return acs.adult_filter(data)

    sensitive_attribute = 'RAC1P'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=general_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()
    dataframe = dataframe[dataframe[sensitive_attribute].isin([1, 2])]
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x > 50000 else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 1 else 0)

    # #plot histogram 'WKHP':
    # dataframe['WKHP'].hist(bins=10)
    # plt.show()



    # positive_rate for A = 1 and A = 0
    positive_rate_A1 = dataframe[(dataframe[sensitive_attribute] == 1) & (dataframe[general_target] == 1)].shape[0] / \
                       dataframe[dataframe[sensitive_attribute] == 1].shape[0]
    positive_rate_A0 = dataframe[(dataframe[sensitive_attribute] == 0) & (dataframe[general_target] == 1)].shape[0] / \
                       dataframe[dataframe[sensitive_attribute] == 0].shape[0]

    # print(f"Positive rate for A=1: {positive_rate_A1}")
    # print(f"Positive rate for A=0: {positive_rate_A0}")
    max_bin = {'COW': 8, 'SEX': 2, 'WKHP': 10, 'SCHL': 24}
    features_to_use = [feature for feature in dataframe.columns if
                       feature != general_target and feature != sensitive_attribute]
    if all:
        # features_to_use = [MAR, PINCP, ESR, ST, NATIVITY, DREM]
        features_to_use = ['COW', 'SEX', 'WKHP', 'SCHL']
        categorical_features = ['COW', 'SEX', 'SCHL']

        for feature in dataframe.columns:
            if feature not in features_to_use and feature != general_target and feature != sensitive_attribute:
                dataframe.drop(feature, axis=1, inplace=True)

        # print(dataframe.columns)

        number_of_unique_values = {}
        for feature in dataframe.columns:
            if feature != general_target and feature != sensitive_attribute:
                if feature in categorical_features:
                    number_of_unique_values[feature] = dataframe[feature].nunique()
                else:
                    number_of_unique_values[feature] = 100

        # print(number_of_unique_values)
        bins = {}
        for feature in number_of_unique_values.keys():
            bins[feature] = min(max_bin[feature], number_of_unique_values[feature])

        for feature in categorical_features:
            if number_of_unique_values[feature] > bins[feature]:
                dataframe[feature] = dataframe[feature].astype('category')
                target_encoder = TargetEncoder()
                dataframe[feature + '_encoded'] = target_encoder.fit_transform(dataframe[feature],
                                                                               dataframe[general_target])
                occ_index = dataframe.columns.get_loc(feature)
                dataframe.drop(feature, axis=1, inplace=True)
                dataframe.insert(occ_index, feature + '_encoded', dataframe.pop(feature + '_encoded'))
                dataframe.rename(columns={feature + '_encoded': feature}, inplace=True)
                # dataframe.rename(columns={'SCHL_encoded': 'SCHL'}, inplace=True)

        # print(dataframe.head())
        # apply binning
        for feature in dataframe.columns:
            if feature != general_target and feature != sensitive_attribute:
                dataframe[feature] = pd.cut(dataframe[feature], bins[feature], labels=False)

        bins = tuple([bins[feature] for feature in ['COW', 'SEX', 'WKHP', 'SCHL']] + [2, 2])
        features = ['COW', 'SEX', 'WKHP', 'SCHL']
        output = [general_target]
        sensitive_attribute = [sensitive_attribute]
        # print(features)
        # print(sensitive_attribute)
        # print(output)
        # print(bins)
        # print(dataframe.head())
        return dataframe, features, sensitive_attribute, output, bins


def generate_dataset_folktables_RAC1P_WB_selective_updated(n_samples=10000, parameters=[], seed=0):
    all = True
    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)

    # ACSIncome.features = [ 'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']

    general_features = list(set(ACSIncome.features))

    general_target = ACSIncome.target

    def general_filter(data):
        return acs.adult_filter(data)

    sensitive_attribute = 'RAC1P'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=general_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()
    dataframe = dataframe[dataframe[sensitive_attribute].isin([1, 2])]
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x > 50000 else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 1 else 0)

    # #plot histogram 'WKHP':
    # dataframe['WKHP'].hist(bins=10)
    # plt.show()



    # positive_rate for A = 1 and A = 0
    positive_rate_A1 = dataframe[(dataframe[sensitive_attribute] == 1) & (dataframe[general_target] == 1)].shape[0] / \
                       dataframe[dataframe[sensitive_attribute] == 1].shape[0]
    positive_rate_A0 = dataframe[(dataframe[sensitive_attribute] == 0) & (dataframe[general_target] == 1)].shape[0] / \
                       dataframe[dataframe[sensitive_attribute] == 0].shape[0]

    # print(f"Positive rate for A=1: {positive_rate_A1}")
    # print(f"Positive rate for A=0: {positive_rate_A0}")
    max_bin = {'COW': 8, 'SEX': 2, 'WKHP': 10, 'SCHL': 24}
    features_to_use = [feature for feature in dataframe.columns if
                       feature != general_target and feature != sensitive_attribute]
    if all:
        # features_to_use = [MAR, PINCP, ESR, ST, NATIVITY, DREM]
        features_to_use = ['COW', 'SEX', 'WKHP', 'SCHL']
        categorical_features = ['COW', 'SEX', 'SCHL']

        for feature in dataframe.columns:
            if feature not in features_to_use and feature != general_target and feature != sensitive_attribute:
                dataframe.drop(feature, axis=1, inplace=True)

        # print(dataframe.columns)

        number_of_unique_values = {}
        for feature in dataframe.columns:
            if feature != general_target and feature != sensitive_attribute:
                if feature in categorical_features:
                    number_of_unique_values[feature] = dataframe[feature].nunique()
                else:
                    number_of_unique_values[feature] = 100

        # print(number_of_unique_values)
        bins = {}
        for feature in number_of_unique_values.keys():
            bins[feature] = min(max_bin[feature], number_of_unique_values[feature])

        for feature in categorical_features:
            if number_of_unique_values[feature] > bins[feature]:
                dataframe[feature] = dataframe[feature].astype('category')
                target_encoder = TargetEncoder()
                dataframe[feature + '_encoded'] = target_encoder.fit_transform(dataframe[feature],
                                                                               dataframe[general_target])
                occ_index = dataframe.columns.get_loc(feature)
                dataframe.drop(feature, axis=1, inplace=True)
                dataframe.insert(occ_index, feature + '_encoded', dataframe.pop(feature + '_encoded'))
                dataframe.rename(columns={feature + '_encoded': feature}, inplace=True)
                # dataframe.rename(columns={'SCHL_encoded': 'SCHL'}, inplace=True)

        # print(dataframe.head())
        # apply binning
        continues_features = ['WKHP']
        for feature in dataframe.columns:
            if feature in continues_features:
                dataframe[feature] = pd.qcut(dataframe[feature], q=bins[feature], labels=False, duplicates='drop')
            elif feature != general_target and feature != sensitive_attribute:
                dataframe[feature] = pd.cut(dataframe[feature], bins[feature], labels=False)
        # for feature in dataframe.columns:
        #     if feature != general_target and feature != sensitive_attribute:
        #         dataframe[feature] = pd.cut(dataframe[feature], bins[feature], labels=False)

        bins = tuple([dataframe[feature].nunique() for feature in ['COW', 'SEX', 'WKHP', 'SCHL']] + [2, 2])
        features = ['COW', 'SEX', 'WKHP', 'SCHL']
        output = [general_target]
        sensitive_attribute = [sensitive_attribute]
        # print(features)
        # print(sensitive_attribute)
        # print(output)
        # print(bins)
        # print(dataframe.head())
        return dataframe, features, sensitive_attribute, output, bins, categorical_features

def generate_dataset_folktables_Dis(n_samples=10000, parameters=[], seed=0):
    # To use this function, you need to define the parameters as the number of bins for each feature:
    # features order: ['AGEP', 'SCHL', 'OCCP_encoded', 'MAR_encoded', 'RELP_encoded', 'WKHP'] # MAR_encoded no more that 5 bins
    # Thus, the parameters should be a tuple with 6 elements, each element is the number of bins for each feature

    # example usage:
    # parameters = [10, 10, 10, 5, 10, 10]
    # parameters = [2, 3, 3, 2, 3, 2]
    # df, features, sensitive_attribute, output, bins = generate_dataset_folktables(n_samples=10000, parameters=parameters, seed=0)

    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)

    # ACSIncome.features = [ 'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']

    general_features = list(set(ACSIncome.features))

    general_target = ACSIncome.target

    def general_filter(data):
        return acs.adult_filter(data)

    sensitive_attribute = 'RAC1P'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=general_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()

    ## encoding the categorical features using target encoding
    # OCCP
    dataframe['OCCP'] = dataframe['OCCP'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['OCCP_encoded'] = target_encoder.fit_transform(dataframe['OCCP'], dataframe[general_target])
    occ_index = dataframe.columns.get_loc('OCCP')
    dataframe.drop('OCCP', axis=1, inplace=True)
    dataframe.insert(occ_index, 'OCCP_encoded', dataframe.pop('OCCP_encoded'))
    # MAR
    dataframe['MAR'] = dataframe['MAR'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['MAR_encoded'] = target_encoder.fit_transform(dataframe['MAR'], dataframe[general_target])
    mar_index = dataframe.columns.get_loc('MAR')
    dataframe.drop('MAR', axis=1, inplace=True)
    dataframe.insert(mar_index, 'MAR_encoded', dataframe.pop('MAR_encoded'))
    # RELP
    dataframe['RELP'] = dataframe['RELP'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['RELP_encoded'] = target_encoder.fit_transform(dataframe['RELP'], dataframe[general_target])
    relp_index = dataframe.columns.get_loc('RELP')
    dataframe.drop('RELP', axis=1, inplace=True)
    dataframe.insert(relp_index, 'RELP_encoded', dataframe.pop('RELP_encoded'))

    dataframe[general_target] = dataframe[general_target].apply(lambda x: True if x > 50000 else False)

    # Drop the columns that are not needed: COW, POBP, RAC1P
    dataframe.drop('COW', axis=1, inplace=True)
    dataframe.drop('POBP', axis=1, inplace=True)
    dataframe.drop('SEX', axis=1, inplace=True)
    sensitive_attribute = "RAC1P"
    Target_label = general_target

    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 1 else 0)
    # apply bining to AGE, SCHL, OCCP_encoded, MAR_encoded, RELP_encoded, WKHP
    dataframe['AGEP'] = pd.cut(dataframe['AGEP'], bins=parameters[0], labels=False)
    dataframe['SCHL'] = pd.cut(dataframe['SCHL'], bins=parameters[1], labels=False)
    dataframe['OCCP_encoded'] = pd.cut(dataframe['OCCP_encoded'], bins=parameters[2], labels=False)
    dataframe['MAR_encoded'] = pd.cut(dataframe['MAR_encoded'], bins=parameters[3], labels=False)
    dataframe['RELP_encoded'] = pd.cut(dataframe['RELP_encoded'], bins=parameters[4], labels=False)
    dataframe['WKHP'] = pd.cut(dataframe['WKHP'], bins=parameters[5], labels=False)

    features = ['AGEP', 'SCHL', 'OCCP_encoded', 'MAR_encoded', 'RELP_encoded', 'WKHP']
    bins = (parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5],2,2)
    output = [Target_label]
    sensitive_attribute = [sensitive_attribute]

    return dataframe, features, sensitive_attribute, output, bins

def generate_dataset_Public_Coverage(n_samples=10000, parameters=[], seed=0):
    # parameters = [10, 7, 5, 2, 2, 10] # max_bins
    # parameters = [2, 3, 2, 2, 2, 2] # for shared bins
    # order_of_features = ['SCHL', 'ESR_encoded', 'MAR_encoded', 'DREM', 'DEYE', 'PINCP']
    # I will drop ESP_encoded
    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)


    general_features = list(set(ACSPublicCoverage.features))

    general_target = ACSPublicCoverage.target

    def general_filter(data):
        return acs.public_coverage_filter(data)

    sensitive_attribute = 'DIS'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=general_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x == 1 else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 2 else 0)
    # print(dataframe[general_target].value_counts())

    # ## encoding the categorical features using target encoding
    # # ESP
    # dataframe['ESP'] = dataframe['ESP'].astype('category')
    # target_encoder = TargetEncoder()
    # dataframe['ESP_encoded'] = target_encoder.fit_transform(dataframe['ESP'], dataframe[general_target])
    # occ_index = dataframe.columns.get_loc('ESP')
    # dataframe.drop('ESP', axis=1, inplace=True)
    # dataframe.insert(occ_index, 'ESP_encoded', dataframe.pop('ESP_encoded'))
    # # MAR
    dataframe['MAR'] = dataframe['MAR'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['MAR_encoded'] = target_encoder.fit_transform(dataframe['MAR'], dataframe[general_target])
    mar_index = dataframe.columns.get_loc('MAR')
    dataframe.drop('MAR', axis=1, inplace=True)
    dataframe.insert(mar_index, 'MAR_encoded', dataframe.pop('MAR_encoded'))
    #  # ESR
    dataframe['ESR'] = dataframe['ESR'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['ESR_encoded'] = target_encoder.fit_transform(dataframe['ESR'], dataframe[general_target])
    esr_index = dataframe.columns.get_loc('ESR')
    dataframe.drop('ESR', axis=1, inplace=True)
    dataframe.insert(esr_index, 'ESR_encoded', dataframe.pop('ESR_encoded'))


    order_of_features = ['SCHL', 'ESR_encoded', 'MAR_encoded', 'DREM', 'DEYE', 'PINCP']

    # dataframe['ESP_encoded'] = pd.cut(dataframe['ESP_encoded'], bins=parameters[0], labels=False)
    dataframe['SCHL'] = pd.cut(dataframe['SCHL'], bins=parameters[0], labels=False)
    dataframe['ESR_encoded'] = pd.cut(dataframe['ESR_encoded'], bins=parameters[1], labels=False)
    dataframe['MAR_encoded'] = pd.cut(dataframe['MAR_encoded'], bins=parameters[2], labels=False)
    dataframe['DREM'] = pd.cut(dataframe['DREM'], bins=parameters[3], labels=False)
    dataframe['DEYE'] = pd.cut(dataframe['DEYE'], bins=parameters[4], labels=False)
    dataframe['PINCP'] = pd.cut(dataframe['PINCP'], bins=parameters[5], labels=False)

    # drop RAC1P, DEAR, ANC, NATIVITY, MIG, MIL, FER, ST, SEX, CIT, AGEP
    dataframe.drop('RAC1P', axis=1, inplace=True)
    dataframe.drop('DEAR', axis=1, inplace=True)
    dataframe.drop('ANC', axis=1, inplace=True)
    dataframe.drop('NATIVITY', axis=1, inplace=True)
    dataframe.drop('MIG', axis=1, inplace=True)
    dataframe.drop('MIL', axis=1, inplace=True)
    dataframe.drop('FER', axis=1, inplace=True)
    dataframe.drop('ST', axis=1, inplace=True)
    dataframe.drop('SEX', axis=1, inplace=True)
    dataframe.drop('CIT', axis=1, inplace=True)
    dataframe.drop('AGEP', axis=1, inplace=True)
    dataframe.drop('ESP', axis=1, inplace=True) # new lines
    # order_of_features = ['SCHL', 'ESR_encoded', 'MAR_encoded', 'DREM', 'DEYE', 'PINCP'] # new lines

    features = order_of_features
    bins = (parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], 2, 2)
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]

    return dataframe, features, sensitive_attribute, output, bins

def generate_dataset_Employment(n_samples=10000, parameters=[], seed=0):
    # parameters = [5, 10, 2, 10, 10, 2, 2]  # max_bins
    # parameters = [3, 3, 2, 3, 2, 2, 2]  # min
    # order_of_features = ['MAR_encoded', 'RELP_encoded', 'DREM', 'AGEP', 'SCHL', 'DEAR', 'DIS']

    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)


    general_features = list(set(ACSEmployment.features))

    general_target = ACSEmployment.target

    def general_filter(data):
        return acs.employment_filter(data)

    sensitive_attribute = 'SEX'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=general_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x == 1 else 0)

    # print(dataframe[general_target].value_counts())

    # # ## encoding the categorical features using target encoding
    # # MAR
    dataframe['MAR'] = dataframe['MAR'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['MAR_encoded'] = target_encoder.fit_transform(dataframe['MAR'], dataframe[general_target])
    mar_index = dataframe.columns.get_loc('MAR')
    dataframe.drop('MAR', axis=1, inplace=True)
    dataframe.insert(mar_index, 'MAR_encoded', dataframe.pop('MAR_encoded'))
    # # RELP
    dataframe['RELP'] = dataframe['RELP'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['RELP_encoded'] = target_encoder.fit_transform(dataframe['RELP'], dataframe[general_target])
    relp_index = dataframe.columns.get_loc('RELP')
    dataframe.drop('RELP', axis=1, inplace=True)
    dataframe.insert(relp_index, 'RELP_encoded', dataframe.pop('RELP_encoded'))

    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 2 else 0)
    # parameters = [5, 10, 2, 10, 10, 2, 2]  # max_bins
    order_of_features = ['MAR_encoded', 'RELP_encoded', 'DREM', 'AGEP', 'SCHL', 'DEAR', 'DIS']
    dataframe['MAR_encoded'] = pd.cut(dataframe['MAR_encoded'], bins=parameters[0], labels=False)
    dataframe['RELP_encoded'] = pd.cut(dataframe['RELP_encoded'], bins=parameters[1], labels=False)
    dataframe['DREM'] = pd.cut(dataframe['DREM'], bins=parameters[2], labels=False)
    dataframe['AGEP'] = pd.cut(dataframe['AGEP'], bins=parameters[3], labels=False)
    dataframe['SCHL'] = pd.cut(dataframe['SCHL'], bins=parameters[4], labels=False)
    dataframe['DEAR'] = pd.cut(dataframe['DEAR'], bins=parameters[5], labels=False)
    dataframe['DIS'] = pd.cut(dataframe['DIS'], bins=parameters[6], labels=False)

    # drop CIT, RAC1P, DEYE, MIG, MIL, ANC, ESP, NATIVITY
    dataframe.drop('CIT', axis=1, inplace=True)
    dataframe.drop('RAC1P', axis=1, inplace=True)
    dataframe.drop('DEYE', axis=1, inplace=True)
    dataframe.drop('MIG', axis=1, inplace=True)
    dataframe.drop('MIL', axis=1, inplace=True)
    dataframe.drop('ANC', axis=1, inplace=True)
    dataframe.drop('ESP', axis=1, inplace=True)
    dataframe.drop('NATIVITY', axis=1, inplace=True)
    features = order_of_features
    bins = (parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], 2, 2)
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]

    return dataframe, features, sensitive_attribute, output, bins

def generate_dataset_Employment_selective(n_samples=10000, parameters=[], seed=0):
    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)

    # ACSIncome.features = [ 'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']

    general_features = list(set(ACSEmployment.features))

    general_target = ACSEmployment.target

    def general_filter(data):
        return acs.employment_filter(data)

    sensitive_attribute = 'DIS'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=general_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x == 1 else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 1 else 0)

    # # plot histogram of the AGEP
    # dataframe['AGEP'].hist(bins=20)
    # plt.show()

    # features_to_use = [MAR, PINCP, ESR, ST, NATIVITY, DREM]
    features_to_use = ['RAC1P', 'MAR', 'AGEP', 'MIG', 'ESP']
    categorical_features = ['RAC1P', 'MAR', 'MIG', 'ESP']

    for feature in dataframe.columns:
        if feature not in features_to_use and feature != general_target and feature != sensitive_attribute:
            dataframe.drop(feature, axis=1, inplace=True)

    # print(dataframe.columns)

    number_of_unique_values = {}
    for feature in dataframe.columns:
        if feature != general_target and feature != sensitive_attribute:
            if feature in categorical_features:
                number_of_unique_values[feature] = dataframe[feature].nunique()
            else:
                number_of_unique_values[feature] = 100

    # print(number_of_unique_values)
    bins = {}
    for feature in number_of_unique_values.keys():
        bins[feature] = min(5, number_of_unique_values[feature])

    for feature in categorical_features:
        if number_of_unique_values[feature] > bins[feature]:
            dataframe[feature] = dataframe[feature].astype('category')
            target_encoder = TargetEncoder()
            dataframe[feature + '_encoded'] = target_encoder.fit_transform(dataframe[feature],
                                                                           dataframe[general_target])
            occ_index = dataframe.columns.get_loc(feature)
            dataframe.drop(feature, axis=1, inplace=True)
            dataframe.insert(occ_index, feature + '_encoded', dataframe.pop(feature + '_encoded'))
            dataframe.rename(columns={feature + '_encoded': feature}, inplace=True)
            # dataframe.rename(columns={'SCHL_encoded': 'SCHL'}, inplace=True)

    # apply binning
    for feature in dataframe.columns:
        if feature != general_target and feature != sensitive_attribute:
            dataframe[feature] = pd.cut(dataframe[feature], bins[feature], labels=False)
    features_selected = ['MIG', 'MAR', 'ESP', 'RAC1P', 'AGEP']
    bins = tuple([bins[feature] for feature in features_selected]+[2, 2])
    # features = [feature for feature in dataframe.columns if feature != general_target and feature != sensitive_attribute]
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]
    # print(features)
    # print(sensitive_attribute)
    # print(output)
    # print(bins)
    # print(dataframe.head())
    return dataframe, features_selected, sensitive_attribute, output, bins


def generate_dataset_Employment_selective_updated(n_samples=10000, parameters=[], seed=0):
    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)

    # ACSIncome.features = [ 'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']

    general_features = list(set(ACSEmployment.features))

    general_target = ACSEmployment.target

    def general_filter(data):
        return acs.employment_filter(data)

    sensitive_attribute = 'DIS'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=general_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x == 1 else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 1 else 0)

    # # plot histogram of the AGEP
    # dataframe['AGEP'].hist(bins=20)
    # plt.show()

    # features_to_use = [MAR, PINCP, ESR, ST, NATIVITY, DREM]
    features_to_use = ['RAC1P', 'MAR', 'AGEP', 'MIG', 'ESP']
    categorical_features = ['RAC1P', 'MAR', 'MIG', 'ESP']

    for feature in dataframe.columns:
        if feature not in features_to_use and feature != general_target and feature != sensitive_attribute:
            dataframe.drop(feature, axis=1, inplace=True)

    # print(dataframe.columns)

    number_of_unique_values = {}
    for feature in dataframe.columns:
        if feature != general_target and feature != sensitive_attribute:
            if feature in categorical_features:
                number_of_unique_values[feature] = dataframe[feature].nunique()
            else:
                number_of_unique_values[feature] = 100

    # print(number_of_unique_values)
    bins = {}
    for feature in number_of_unique_values.keys():
        bins[feature] = min(10, number_of_unique_values[feature])

    # for feature in categorical_features:
    #     if number_of_unique_values[feature] > bins[feature]:
    #         dataframe[feature] = dataframe[feature].astype('category')
    #         target_encoder = TargetEncoder()
    #         dataframe[feature + '_encoded'] = target_encoder.fit_transform(dataframe[feature],
    #                                                                        dataframe[general_target])
    #         occ_index = dataframe.columns.get_loc(feature)
    #         dataframe.drop(feature, axis=1, inplace=True)
    #         dataframe.insert(occ_index, feature + '_encoded', dataframe.pop(feature + '_encoded'))
    #         dataframe.rename(columns={feature + '_encoded': feature}, inplace=True)
    #         # dataframe.rename(columns={'SCHL_encoded': 'SCHL'}, inplace=True)

    # apply binning

    continues_features = ['AGEP']
    for feature in dataframe.columns:
        if feature in continues_features:
            dataframe[feature] = pd.qcut(dataframe[feature], q=bins[feature], labels=False)
        # elif feature != general_target and feature != sensitive_attribute:
        #     dataframe[feature] = pd.cut(dataframe[feature], bins[feature], labels=False)

    # for feature in dataframe.columns:
    #     if feature != general_target and feature != sensitive_attribute:
    #         dataframe[feature] = pd.cut(dataframe[feature], bins[feature], labels=False)
    features_selected = ['MIG', 'MAR', 'ESP', 'RAC1P', 'AGEP']
    bins = tuple([dataframe[feature].nunique() for feature in features_selected]+[2, 2])
    # features = [feature for feature in dataframe.columns if feature != general_target and feature != sensitive_attribute]
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]
    # print(features)
    # print(sensitive_attribute)
    # print(output)
    # print(bins)
    # print(dataframe.head())
    return dataframe, features_selected, sensitive_attribute, output, bins, categorical_features

def generate_dataset_Employment_dis(n_samples=10000, parameters=[], seed=0):
    # parameters = [5, 10, 2, 10, 10, 2]  # max_bins
    # parameters = [3, 3, 2, 3, 2, 2]  # min
    # order_of_features = ['MAR_encoded', 'RELP_encoded', 'DREM', 'AGEP', 'SCHL', 'DEAR']

    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)


    general_features = list(set(ACSEmployment.features))

    general_target = ACSEmployment.target

    def general_filter(data):
        return acs.employment_filter(data)

    sensitive_attribute = 'DIS'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=general_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x == 1 else 0)

    # print(dataframe[general_target].value_counts())

    # # ## encoding the categorical features using target encoding
    # # MAR
    dataframe['MAR'] = dataframe['MAR'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['MAR_encoded'] = target_encoder.fit_transform(dataframe['MAR'], dataframe[general_target])
    mar_index = dataframe.columns.get_loc('MAR')
    dataframe.drop('MAR', axis=1, inplace=True)
    dataframe.insert(mar_index, 'MAR_encoded', dataframe.pop('MAR_encoded'))
    # # RELP
    dataframe['RELP'] = dataframe['RELP'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['RELP_encoded'] = target_encoder.fit_transform(dataframe['RELP'], dataframe[general_target])
    relp_index = dataframe.columns.get_loc('RELP')
    dataframe.drop('RELP', axis=1, inplace=True)
    dataframe.insert(relp_index, 'RELP_encoded', dataframe.pop('RELP_encoded'))

    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 2 else 0)
    # parameters = [5, 10, 2, 10, 10, 2, 2]  # max_bins
    order_of_features = ['MAR_encoded', 'RELP_encoded', 'DREM', 'AGEP', 'SCHL', 'DEAR']
    dataframe['MAR_encoded'] = pd.cut(dataframe['MAR_encoded'], bins=parameters[0], labels=False)
    dataframe['RELP_encoded'] = pd.cut(dataframe['RELP_encoded'], bins=parameters[1], labels=False)
    dataframe['DREM'] = pd.cut(dataframe['DREM'], bins=parameters[2], labels=False)
    dataframe['AGEP'] = pd.cut(dataframe['AGEP'], bins=parameters[3], labels=False)
    dataframe['SCHL'] = pd.cut(dataframe['SCHL'], bins=parameters[4], labels=False)
    dataframe['DEAR'] = pd.cut(dataframe['DEAR'], bins=parameters[5], labels=False)
    # dataframe['DIS'] = pd.cut(dataframe['DIS'], bins=parameters[6], labels=False)

    # drop CIT, RAC1P, DEYE, MIG, MIL, ANC, ESP, NATIVITY
    dataframe.drop('CIT', axis=1, inplace=True)
    dataframe.drop('RAC1P', axis=1, inplace=True)
    dataframe.drop('DEYE', axis=1, inplace=True)
    dataframe.drop('MIG', axis=1, inplace=True)
    dataframe.drop('MIL', axis=1, inplace=True)
    dataframe.drop('ANC', axis=1, inplace=True)
    dataframe.drop('ESP', axis=1, inplace=True)
    dataframe.drop('NATIVITY', axis=1, inplace=True)
    dataframe.drop('SEX', axis=1, inplace=True)
    features = order_of_features
    bins = (parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], 2, 2)
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]

    return dataframe, features, sensitive_attribute, output, bins


def generate_dataset_Mobility(n_samples=10000, parameters=[], seed=0):
    # order_of_features = ['MIL_encoded', 'ESR_encoded', 'PINCP', 'AGEP', 'MAR_encoded', 'RELP_encoded']
    # parameters = [4, 6, 10, 10, 5, 10]  # max_bins
    # parameters = [3, 2, 2, 2, 2, 3]  # min_bins
    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)

    general_features = list(set(ACSMobility.features))

    general_target = ACSMobility.target

    sensitive_attribute = 'GCL'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x == 1,
        group=sensitive_attribute,
        preprocess=lambda x: x.drop(x.loc[(x['AGEP'] <= 18) | (x['AGEP'] >= 35)].index),
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x == 1 else 0)

    # print(dataframe[general_target].value_counts())
    # print(dataframe.head(10))

    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 2 else 0)

    # MIL, ESR, RELP, MAR
    ## MAR
    # print(dataframe['MAR'].value_counts())
    dataframe['MAR'] = dataframe['MAR'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['MAR_encoded'] = target_encoder.fit_transform(dataframe['MAR'], dataframe[general_target])
    mar_index = dataframe.columns.get_loc('MAR')
    dataframe.drop('MAR', axis=1, inplace=True)
    dataframe.insert(mar_index, 'MAR_encoded', dataframe.pop('MAR_encoded'))
    ## RELP
    # print(dataframe['RELP'].value_counts())
    dataframe['RELP'] = dataframe['RELP'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['RELP_encoded'] = target_encoder.fit_transform(dataframe['RELP'], dataframe[general_target])
    relp_index = dataframe.columns.get_loc('RELP')
    dataframe.drop('RELP', axis=1, inplace=True)
    dataframe.insert(relp_index, 'RELP_encoded', dataframe.pop('RELP_encoded'))
    ## ESR
    # print(dataframe['ESR'].value_counts())
    dataframe['ESR'] = dataframe['ESR'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['ESR_encoded'] = target_encoder.fit_transform(dataframe['ESR'], dataframe[general_target])
    esr_index = dataframe.columns.get_loc('ESR')
    dataframe.drop('ESR', axis=1, inplace=True)
    ## MIL
    # print(dataframe['MIL'].value_counts())
    dataframe['MIL'] = dataframe['MIL'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['MIL_encoded'] = target_encoder.fit_transform(dataframe['MIL'], dataframe[general_target])
    mil_index = dataframe.columns.get_loc('MIL')
    dataframe.drop('MIL', axis=1, inplace=True)
    dataframe.insert(mil_index, 'MIL_encoded', dataframe.pop('MIL_encoded'))

    # MIL_encoded, ESR_encoded, PINCP_encoded, AGEP, MAR_encoded, RELP_encoded
    order_of_features = ['MIL_encoded', 'ESR_encoded', 'PINCP', 'AGEP', 'MAR_encoded', 'RELP_encoded']
    # parameters = [4, 6, 10, 10, 5, 10]  # max_bins
    dataframe['MAR_encoded'] = pd.cut(dataframe['MAR_encoded'], bins=parameters[0], labels=False)
    dataframe['RELP_encoded'] = pd.cut(dataframe['RELP_encoded'], bins=parameters[1], labels=False)
    dataframe['PINCP'] = pd.cut(dataframe['PINCP'], bins=parameters[2], labels=False)
    dataframe['AGEP'] = pd.cut(dataframe['AGEP'], bins=parameters[3], labels=False)
    dataframe['MIL_encoded'] = pd.cut(dataframe['MIL_encoded'], bins=parameters[4], labels=False)
    dataframe['ESR_encoded'] = pd.cut(dataframe['ESR_encoded'], bins=parameters[5], labels=False)

    # drop DIS, DEYE, SEX, ESR, NATIVITY, COW, WKHP, RAC1P, SCHL, JWMNP, CIT, DREM, ANC, DEAR
    dataframe.drop('DIS', axis=1, inplace=True)
    dataframe.drop('DEYE', axis=1, inplace=True)
    dataframe.drop('SEX', axis=1, inplace=True)
    dataframe.drop('ESP', axis=1, inplace=True)
    dataframe.drop('NATIVITY', axis=1, inplace=True)
    dataframe.drop('COW', axis=1, inplace=True)
    dataframe.drop('WKHP', axis=1, inplace=True)
    dataframe.drop('RAC1P', axis=1, inplace=True)
    dataframe.drop('SCHL', axis=1, inplace=True)
    dataframe.drop('JWMNP', axis=1, inplace=True)
    dataframe.drop('CIT', axis=1, inplace=True)
    dataframe.drop('DREM', axis=1, inplace=True)
    dataframe.drop('ANC', axis=1, inplace=True)
    dataframe.drop('DEAR', axis=1, inplace=True)
    features = order_of_features

    bins = (parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], 2, 2)
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]

    return dataframe, features, sensitive_attribute, output, bins


def generate_dataset_ACSTravelTime_selective_new(n_samples=10000, parameters=[], seed=0):
    all = True
    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)

    # ACSIncome.features = [ 'AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']

    general_features = list(set(ACSTravelTime.features))

    general_target = ACSTravelTime.target

    sensitive_attribute = 'SEX'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=acs.travel_time_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()
    # dataframe = dataframe[dataframe[sensitive_attribute].isin([1, 2])]
    # dataframe = dataframe[dataframe[sensitive_attribute].isin([1, 2])]
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x > 20 else 0)
    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 1 else 0)
    # positive_rate for A = 1 and A = 0
    positive_rate_A1 = dataframe[(dataframe[sensitive_attribute] == 1) & (dataframe[general_target] == 1)].shape[0] / \
                       dataframe[dataframe[sensitive_attribute] == 1].shape[0]
    positive_rate_A0 = dataframe[(dataframe[sensitive_attribute] == 0) & (dataframe[general_target] == 1)].shape[0] / \
                       dataframe[dataframe[sensitive_attribute] == 0].shape[0]

    # print(f"Positive rate for A=1: {positive_rate_A1}")
    # print(f"Positive rate for A=0: {positive_rate_A0}")
    features_to_use = [feature for feature in dataframe.columns if
                       feature != general_target and feature != sensitive_attribute]
    if all:
        # features_to_use = [MAR, PINCP, ESR, ST, NATIVITY, DREM]
        features_to_use = ['ESP', 'RELP', 'POWPUMA', 'DIS', 'AGEP', 'MIG', 'PUMA', 'CIT',
                           'JWTR', 'SCHL', 'POVPIP', 'RAC1P', 'OCCP', 'ST', 'MAR']
        categorical_features = ['ESP', 'RELP', 'POWPUMA', 'DIS', 'MIG', 'PUMA', 'CIT',
                                'JWTR', 'SCHL', 'RAC1P', 'OCCP', 'ST', 'MAR']

        categorical_features = [feature for feature in categorical_features if feature in features_to_use]

        for feature in dataframe.columns:
            if feature not in features_to_use and feature != general_target and feature != sensitive_attribute:
                dataframe.drop(feature, axis=1, inplace=True)

        # print(dataframe.columns)

        number_of_unique_values = {}
        for feature in dataframe.columns:
            if feature != general_target and feature != sensitive_attribute:
                if feature in categorical_features:
                    number_of_unique_values[feature] = dataframe[feature].nunique()
                else:
                    number_of_unique_values[feature] = 100

        # print(number_of_unique_values)
        bins = {}
        for feature in number_of_unique_values.keys():
            bins[feature] = min(5, number_of_unique_values[feature])

        for feature in categorical_features:
            dataframe[feature] = dataframe[feature].astype('category')
            target_encoder = TargetEncoder()
            dataframe[feature + '_encoded'] = target_encoder.fit_transform(dataframe[feature],
                                                                           dataframe[general_target])
            occ_index = dataframe.columns.get_loc(feature)
            dataframe.drop(feature, axis=1, inplace=True)
            dataframe.insert(occ_index, feature + '_encoded', dataframe.pop(feature + '_encoded'))
            dataframe.rename(columns={feature + '_encoded': feature}, inplace=True)
            # dataframe.rename(columns={'SCHL_encoded': 'SCHL'}, inplace=True)

        # print(dataframe.head())
        # apply binning
        for feature in dataframe.columns:
            if feature != general_target and feature != sensitive_attribute:
                dataframe[feature] = pd.cut(dataframe[feature], bins[feature], labels=False)
    #
    for feature in features_to_use:
        bins[feature] = dataframe[feature].nunique()
    # for feature in dataframe.columns:
    #     print(feature, dataframe[feature].unique())

    bins = tuple(list(bins.values())+ [2, 2])
    features = features_to_use
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]

    return dataframe, features, sensitive_attribute, output, bins

def generate_dataset_ACSTravelTime(n_samples=10000, parameters=[], seed=0):
    # order_of_features = ['AGEP', 'POWPUMA_encoded', 'OCCP_encoded', 'JWTR_encoded', 'PUMA_encoded', 'ST_encoded']
    # parameters = [10, 10, 10, 10, 10, 10]  # max_bins
    # parameters = [2, 3, 2, 3, 2, 2]  # min_bins
    state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    # state_list = ['MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=state_list, download=True)

    general_features = list(set(ACSTravelTime.features))

    general_target = ACSTravelTime.target

    sensitive_attribute = 'SEX'

    New_Task = BasicProblem(
        features=general_features,
        target=general_target,
        target_transform=lambda x: x,
        group=sensitive_attribute,
        preprocess=acs.travel_time_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    df = New_Task.df_to_pandas(acs_data)

    features_df = df[0]  # It includes the senstive attribute
    if general_target in features_df.columns:
        features_df.drop(general_target, axis=1, inplace=True)

    target_df = df[1]
    dataframe = pd.concat([features_df, target_df], axis=1)
    dataframe.replace(-1, np.nan, inplace=True)
    dataframe = dataframe.dropna()
    dataframe[general_target] = dataframe[general_target].apply(lambda x: 1 if x > 20 else 0)

    # print(dataframe[general_target].value_counts())
    # print(dataframe.head(10))

    dataframe[sensitive_attribute] = dataframe[sensitive_attribute].apply(lambda x: 1 if x == 2 else 0)

    # to be used: AGEP, POWPUMA, OCCP, JWTR, PUMA, ST
    # categorical_features = ['POWPUMA', 'OCCP', 'JWTR', 'PUMA', 'ST']
    ## POWPUMA
    # print(dataframe['POWPUMA'].value_counts())
    dataframe['POWPUMA'] = dataframe['POWPUMA'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['POWPUMA_encoded'] = target_encoder.fit_transform(dataframe['POWPUMA'], dataframe[general_target])
    powpuma_index = dataframe.columns.get_loc('POWPUMA')
    dataframe.drop('POWPUMA', axis=1, inplace=True)
    dataframe.insert(powpuma_index, 'POWPUMA_encoded', dataframe.pop('POWPUMA_encoded'))
    ## OCCP
    # print(dataframe['OCCP'].value_counts())
    dataframe['OCCP'] = dataframe['OCCP'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['OCCP_encoded'] = target_encoder.fit_transform(dataframe['OCCP'], dataframe[general_target])
    occp_index = dataframe.columns.get_loc('OCCP')
    dataframe.drop('OCCP', axis=1, inplace=True)
    dataframe.insert(occp_index, 'OCCP_encoded', dataframe.pop('OCCP_encoded'))
    ## JWTR
    # print(dataframe['JWTR'].value_counts())
    dataframe['JWTR'] = dataframe['JWTR'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['JWTR_encoded'] = target_encoder.fit_transform(dataframe['JWTR'], dataframe[general_target])
    jwtr_index = dataframe.columns.get_loc('JWTR')
    dataframe.drop('JWTR', axis=1, inplace=True)
    dataframe.insert(jwtr_index, 'JWTR_encoded', dataframe.pop('JWTR_encoded'))
    ## PUMA
    # print(dataframe['PUMA'].value_counts())
    dataframe['PUMA'] = dataframe['PUMA'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['PUMA_encoded'] = target_encoder.fit_transform(dataframe['PUMA'], dataframe[general_target])
    puma_index = dataframe.columns.get_loc('PUMA')
    dataframe.drop('PUMA', axis=1, inplace=True)
    dataframe.insert(puma_index, 'PUMA_encoded', dataframe.pop('PUMA_encoded'))
    ## ST
    # print(dataframe['ST'].value_counts())
    dataframe['ST'] = dataframe['ST'].astype('category')
    target_encoder = TargetEncoder()
    dataframe['ST_encoded'] = target_encoder.fit_transform(dataframe['ST'], dataframe[general_target])
    st_index = dataframe.columns.get_loc('ST')
    dataframe.drop('ST', axis=1, inplace=True)
    dataframe.insert(st_index, 'ST_encoded', dataframe.pop('ST_encoded'))

    order_of_features = ['AGEP', 'POWPUMA_encoded', 'OCCP_encoded', 'JWTR_encoded', 'PUMA_encoded', 'ST_encoded']
    # parameters = [10, 10, 10, 10, 10, 10]  # max_bins
    dataframe['AGEP'] = pd.cut(dataframe['AGEP'], bins=parameters[0], labels=False)
    dataframe['POWPUMA_encoded'] = pd.cut(dataframe['POWPUMA_encoded'], bins=parameters[1], labels=False)
    dataframe['OCCP_encoded'] = pd.cut(dataframe['OCCP_encoded'], bins=parameters[2], labels=False)
    dataframe['JWTR_encoded'] = pd.cut(dataframe['JWTR_encoded'], bins=parameters[3], labels=False)
    dataframe['PUMA_encoded'] = pd.cut(dataframe['PUMA_encoded'], bins=parameters[4], labels=False)
    dataframe['ST_encoded'] = pd.cut(dataframe['ST_encoded'], bins=parameters[5], labels=False)

    # drop: POVPIP, ESP, CIT, MAR, SCHL, RAC1P, RELP, DIS, MIG
    dataframe.drop('POVPIP', axis=1, inplace=True)
    dataframe.drop('ESP', axis=1, inplace=True)
    dataframe.drop('CIT', axis=1, inplace=True)
    dataframe.drop('MAR', axis=1, inplace=True)
    dataframe.drop('SCHL', axis=1, inplace=True)
    dataframe.drop('RAC1P', axis=1, inplace=True)
    dataframe.drop('RELP', axis=1, inplace=True)
    dataframe.drop('DIS', axis=1, inplace=True)
    dataframe.drop('MIG', axis=1, inplace=True)
    features = order_of_features

    bins = (parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], 2, 2)
    output = [general_target]
    sensitive_attribute = [sensitive_attribute]

    return dataframe, features, sensitive_attribute, output, bins

def generate_Canonical_Example_1(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)
    Ux1 = np.random.binomial(1, 0.5, n_samples)
    Ux2 = np.random.binomial(1, 0.5, n_samples)
    Xc = A + Ux1
    Xg = Ux2
    Y = Xc + Xg

    Y = np.where(Y > 1.5, 1, 0)

    if parameters[0] == 1:
        # Generate w to add randomness to Y
        w = np.random.binomial(1, 0.1, n_samples)

        # Flip Y where w is 1
        Y = np.where(w == 1, 1 - Y, Y)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'Xc': Xc,
        'Xg': Xg,
        'A': A,
        'Y': Y
    })
    features = ['Xc', 'Xg']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (2, 2, 2, 2)
    return df, features, sensitive_attribute, output, bins
def generate_small(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)
    X1 = 2*A + np.random.normal(1, 0.5, n_samples)
    X2 = 2*A + np.random.normal(1, 0.5, n_samples)
    Y = X1 + X2

    Y = np.where(Y > 1.5, 1, 0)


    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (10, 10, 2, 2)
    return df, features, sensitive_attribute, output, bins
def generate_Canonical_Example_3(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)
    Ux1 = np.random.binomial(1, 0.5, n_samples)
    Ux2 = np.random.binomial(1, 0.5, n_samples)
    Xc = Ux1 + A
    Xg = Ux1
    Y = Xg

    if parameters[0] == 1:
        # Generate w to add randomness to Y
        w = np.random.binomial(1, 0.1, n_samples)

        # Flip Y where w is 1
        Y = np.where(w == 1, 1 - Y, Y)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'Xc': Xc,
        'Xg': Xg,
        'A': A,
        'Y': Y
    })
    features = ['Xc', 'Xg']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (3, 2, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_Canonical_Example_4(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)
    Ux1 = np.random.binomial(1, 0.5, n_samples)
    Ux2 = np.random.binomial(1, 0.5, n_samples)
    Xc = Ux1
    Xg = A
    Y = Xc^Xg

    if parameters[0] == 1:
        # Generate w to add randomness to Y
        w = np.random.binomial(1, 0.1, n_samples)

        # Flip Y where w is 1
        Y = np.where(w == 1, 1 - Y, Y)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'Xc': Xc,
        'Xg': Xg,
        'A': A,
        'Y': Y
    })
    features = ['Xc', 'Xg']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (2, 2, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_Canonical_Example_6(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)
    Ux1 = np.random.binomial(1, 0.5, n_samples)
    Ux2 = np.random.binomial(1, 0.5, n_samples)
    Xc = Ux1 + A
    Xg = Ux1
    Y = A

    if parameters[0] == 1:
        # Generate w to add randomness to Y
        w = np.random.binomial(1, 0.1, n_samples)

        # Flip Y where w is 1
        Y = np.where(w == 1, 1 - Y, Y)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'Xc': Xc,
        'Xg': Xg,
        'A': A,
        'Y': Y
    })
    features = ['Xc', 'Xg']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (3, 2, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_Exp_Example_1(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)
    sigma = 0.9
    Ux1 = sigma * np.random.normal(0, 1, n_samples)
    Ux2 = sigma * np.random.normal(0, 1, n_samples)
    Ux3 = sigma * np.random.normal(0, 1, n_samples)

    X1 = Ux1 + A
    X2 = Ux2 + A
    X3 = Ux3
    Y = X1 + X2 + X3
    Y = np.where(Y > 1, 1, 0)

    if parameters[0] == 1:
        # Generate w to add randomness to Y
        w = np.random.binomial(1, 0.1, n_samples)

        # Flip Y where w is 1
        Y = np.where(w == 1, 1 - Y, Y)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (5, 5, 5, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_Exp_Example_2(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)
    sigma = 0.5
    Ux1 = np.random.binomial(1, 0.5, n_samples)
    Ux2 = sigma * np.random.normal(0, 1, n_samples)
    Ux3 = sigma * np.random.normal(0, 1, n_samples)

    X1 = Ux1 + Ux3
    X2 = Ux2
    X3 = Ux3
    X4 = Ux2 - A
    Y = (X1 + X4)**2
    Y = np.where(Y > 1, 1, 0)

    if parameters[0] == 1:
        # Generate w to add randomness to Y
        w = np.random.binomial(1, 0.1, n_samples)

        # Flip Y where w is 1
        Y = np.where(w == 1, 1 - Y, Y)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (3, 3, 3, 3, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_Exp_Example_3(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)
    sigma = 0.5
    Ux1 = np.random.binomial(1, 0.5, n_samples)

    Ux2 = sigma * np.random.normal(0, 1, n_samples)
    Ux1_prim = sigma * np.random.normal(0, 1, n_samples)

    X1 = Ux1 + Ux1_prim
    X2 = Ux2
    X3 = Ux2 -A
    Y = (X1 + X3)**2
    Y = np.where(Y > 0.5, 1, 0)

    if parameters[0] == 1:
        # Generate w to add randomness to Y
        w = np.random.binomial(1, 0.1, n_samples)

        # Flip Y where w is 1
        Y = np.where(w == 1, 1 - Y, Y)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (5, 5, 5, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_Exp_Example_4(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)
    sigma = 0.5
    Ux1 = np.random.binomial(1, 0.5, n_samples)

    Ux2 = sigma * np.random.normal(0, 1, n_samples)
    Ux3 = sigma * np.random.normal(0, 1, n_samples)
    Ux1_prim = sigma * np.random.normal(0, 1, n_samples)

    X1 = Ux1 + Ux1_prim + A
    X2 = Ux2
    X3 = Ux3
    X4 = Ux2 + A
    Y = Ux1 + Ux2
    Y = np.where(Y > 0.5, 1, 0)

    if parameters[0] == 1:
        # Generate w to add randomness to Y
        w = np.random.binomial(1, 0.1, n_samples)

        # Flip Y where w is 1
        Y = np.where(w == 1, 1 - Y, Y)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (3, 3, 3, 3, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_model_new_1(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)
    sigma = 0.75
    X1 = (2 * A-1) + np.random.normal(0, sigma, n_samples)
    X2 = (2 * A-1) + np.random.normal(0, sigma, n_samples)

    X3 = -X1 + X2 + np.random.normal(0, sigma, n_samples)
    X4 = 2 * X2 + np.random.normal(0, sigma, n_samples)

    Y =  X3 + X4 +  np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]

    Y = np.where(Y > 0, 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (3, 3, 3, 3, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_model_new_2(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)
    sigma = 0.75
    X1 = (2 * A - 1) + np.random.normal(0, sigma, n_samples)
    X2 = (2 * A - 1) - X1 + np.random.normal(0, sigma, n_samples)

    X3 = - X1 + np.random.normal(0, sigma, n_samples)
    X4 = 2 * X2 + np.random.normal(0, sigma, n_samples)

    Y = X3 + X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]

    Y = np.where(Y > 0, 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (3, 3, 3, 3, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_model_new_3(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)
    sigma = 0.75
    X1 = (2 * A - 1) + np.random.normal(0, sigma, n_samples)
    X2 = (2 * A - 1) - X1 + np.random.normal(0, sigma, n_samples)

    X3 = X1 + np.random.normal(0, sigma, n_samples)
    X4 = 2 * X2 + np.random.normal(0, sigma, n_samples)

    Y = X3 + X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]

    Y = np.where(Y > 0, 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_1(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]
    X1 = (2 * A - 1) + np.random.normal(0, sigma, n_samples)
    X2 = (4 * A - 2) + np.random.normal(0, sigma, n_samples)

    X3 = X1 + X2 + np.random.normal(0, sigma, n_samples)
    X4 = X1 + X2 + np.random.normal(0, sigma, n_samples)

    Y = 2 * X3 + 2 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_2(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]
    X1 = (2 * A - 1) + np.random.normal(0, sigma, n_samples)
    X2 = np.random.normal(0, sigma, n_samples)

    X3 = X1 + X2 + np.random.normal(0, sigma, n_samples)
    X4 = X1 + X2 + np.random.normal(0, sigma, n_samples)

    Y = 2 * X3 + 2 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]

    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_3(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]
    X1 = (2 * A - 1) + np.random.normal(0, sigma, n_samples)
    X2 = np.random.normal(0, sigma, n_samples)

    X3 = X1 + X2 + np.random.normal(0, sigma, n_samples)
    X4 = X2 + np.random.normal(0, sigma, n_samples)

    Y = 2 * X3 + 2 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_4(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]
    X1 = np.random.normal(0, sigma, n_samples)
    X2 = np.random.normal(0, sigma, n_samples)

    X3 = X1 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)
    X4 = X2 + np.random.normal(0, sigma, n_samples) + (4 * A - 2)

    Y = 2 * X3 + 2 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_5(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]
    X1 = np.random.normal(0, sigma, n_samples)
    X2 = np.random.normal(0, sigma, n_samples)

    X3 = X1 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)
    X4 = X2 + np.random.normal(0, sigma, n_samples) + (4 * A - 1)

    Y = 2 * X1 + 1 * X2 + 4 * X3 + 2 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)
    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_6(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]
    X1 = np.random.normal(0, sigma, n_samples) + (2 * A - 1)
    X2 = np.random.normal(0, sigma, n_samples) + (4 * A - 1)

    X3 = X1 + np.random.normal(0, sigma, n_samples)
    X4 = X2 + np.random.normal(0, sigma, n_samples)

    Y = 2 * X1 + 4 * X2 + 4 * X3 + 2 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_7(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]
    X1 = np.random.normal(0, sigma, n_samples) + (A - 0.5)
    X2 = np.random.normal(0, sigma, n_samples) + (2 * A - 1)

    X3 = np.random.normal(0, sigma, n_samples) + (3 * A - 1.5)
    X4 = np.random.normal(0, sigma, n_samples) + (4 * A - 2)

    Y = 2 * X1 + 2 * X2 + 2 * X3 + 2 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_8(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]
    X2 = np.random.normal(0, sigma, n_samples) + (2 * A - 1)

    X3 = np.random.normal(0, sigma, n_samples) + (3 * A - 1.5)
    X4 = np.random.normal(0, sigma, n_samples) + (4 * A - 2)

    X1 = X3 + np.random.normal(0, sigma, n_samples) + (A - 0.5)


    Y = 4 * X1 + 2 * X2 + 2 * X3 + 2 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_1_latent(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)

    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]

    u1 = np.random.normal(0, sigma, n_samples)
    u2 = np.random.normal(0, sigma, n_samples)

    X1 = (2 * A - 1) + np.random.normal(0, sigma, n_samples) + u1
    X2 = (4 * A - 2) + np.random.normal(0, sigma, n_samples) + u2

    X3 = X1 + X2 + np.random.normal(0, sigma, n_samples)
    X4 = X1 + X2 + np.random.normal(0, sigma, n_samples)

    Y = 2 * X3 + 2 * X4 + np.random.normal(0, sigma, n_samples) + u1 + u2 + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_2_latent(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]

    u1 = np.random.normal(0, sigma, n_samples)
    u2 = np.random.normal(0, sigma, n_samples)

    X1 = (2 * A - 1) + np.random.normal(0, sigma, n_samples) + u1
    X2 = np.random.normal(0, sigma, n_samples) + u1

    X3 = X1 + X2 + np.random.normal(0, sigma, n_samples)
    X4 = X1 + X2 + np.random.normal(0, sigma, n_samples)

    Y = 2 * X3 + 2 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_3_latent(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]

    u1 = np.random.normal(0, sigma, n_samples)
    u2 = np.random.normal(0, sigma, n_samples)

    X1 = (2 * A - 1) + np.random.normal(0, sigma, n_samples) + u1
    X2 = np.random.normal(0, sigma, n_samples) + u1

    X3 = X1 + X2 + np.random.normal(0, sigma, n_samples)
    X4 = X2 + np.random.normal(0, sigma, n_samples)

    Y = 2 * X3 + 2 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_4_latent(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]


    u1 = np.random.normal(0, sigma, n_samples)
    u2 = np.random.normal(0, sigma, n_samples)


    X1 = np.random.normal(0, sigma, n_samples) + u1
    X2 = np.random.normal(0, sigma, n_samples) + u2

    X3 = X1 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)
    X4 = X2 + np.random.normal(0, sigma, n_samples) + (4 * A - 2)

    Y = 2 * X3 + 2 * X4 + np.random.normal(0, sigma, n_samples) + u1 + u2 + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_5_latent(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]

    u1 = np.random.normal(0, sigma, n_samples)
    u2 = np.random.normal(0, sigma, n_samples)

    X1 = np.random.normal(0, sigma, n_samples) + u1
    X2 = np.random.normal(0, sigma, n_samples) + u1

    X3 = X1 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)
    X4 = X2 + np.random.normal(0, sigma, n_samples) + (4 * A - 1)

    Y = 2 * X1 + 1 * X2 + 4 * X3 + 2 * X4 + np.random.normal(0, sigma, n_samples) + u1 + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_6_latent(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]

    u1 = np.random.normal(0, sigma, n_samples)
    u2 = np.random.normal(0, sigma, n_samples)

    X1 = np.random.normal(0, sigma, n_samples) + (2 * A - 1)
    X2 = np.random.normal(0, sigma, n_samples) + (4 * A - 1)

    X3 = X1 + np.random.normal(0, sigma, n_samples) + u1
    X4 = X2 + np.random.normal(0, sigma, n_samples) + u1

    Y = 2 * X1 + 4 * X2 + 4 * X3 + 2 * X4 + np.random.normal(0, sigma, n_samples) + u1 + (2 * A - 1)*parameters[0]

    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_7_latent(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]

    u1 = np.random.normal(0, sigma, n_samples)
    u2 = np.random.normal(0, sigma, n_samples)

    X1 = np.random.normal(0, sigma, n_samples) + (A - 0.5) + u1 + u2
    X2 = np.random.normal(0, sigma, n_samples) + (2 * A - 1)

    X3 = np.random.normal(0, sigma, n_samples) + (3 * A - 1.5) + u1
    X4 = np.random.normal(0, sigma, n_samples) + (4 * A - 2) + u2

    Y = 2 * X1 + 2 * X2 + 2 * X3 + 2 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]

    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_8_latent(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]

    u1 = np.random.normal(0, sigma, n_samples)
    u2 = np.random.normal(0, sigma, n_samples)

    X2 = np.random.normal(0, sigma, n_samples) + (2 * A - 1) + u1

    X3 = np.random.normal(0, sigma, n_samples) + (3 * A - 1.5)
    X4 = np.random.normal(0, sigma, n_samples) + (4 * A - 2) + u1

    X1 = X3 + np.random.normal(0, sigma, n_samples) + (A - 0.5)

    Y = 4 * X1 + 2 * X2 + 2 * X3 + 2 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]

    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_model_subsets_design_1(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]

    X1 = np.random.normal(0, sigma, n_samples) + 2 * (A - 0.5)
    X2 = np.random.normal(0, sigma, n_samples) + 4 * (A - 0.5)

    X3 = np.random.normal(0, sigma, n_samples)
    X4 = np.random.normal(0, sigma, n_samples)

    Y = 0.5 * X1 + 0.25 * X2 + 2 * X3 + 4 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y/np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_model_subsets_design_2(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]

    X1 = np.random.normal(0, sigma, n_samples) + 2 * (A - 0.5)
    X2 = np.random.normal(0, sigma, n_samples) + 4 * (A - 0.5)

    X3 = np.random.normal(0, sigma, n_samples) + X2
    X4 = np.random.normal(0, sigma, n_samples)

    Y = 0.5 * X1 + 0.25 * X2 + 2 * X3 + 4 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_model_subsets_design_3(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]

    X1 = np.random.normal(0, sigma, n_samples) + 0.5 * (A - 0.5)
    X2 = np.random.normal(0, sigma, n_samples) + 0.25 * (A - 0.5)

    X3 = np.random.normal(0, sigma, n_samples)
    X4 = np.random.normal(0, sigma, n_samples)

    Y = 2 * X1 + 4 * X2 + 2 * X3 + 4 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_model_subsets_design_4(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]

    X1 = np.random.normal(0, sigma, n_samples) + 0.5 * (A - 0.5)
    X2 = np.random.normal(0, sigma, n_samples) + 0.25 * (A - 0.5)

    X3 = np.random.normal(0, sigma, n_samples) + X2
    X4 = np.random.normal(0, sigma, n_samples)

    Y = 2 * X1 + 4 * X2 + 2 * X3 + 4 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_model_subsets_design_5(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]

    X1 = np.random.normal(0, sigma, n_samples) + 0.5 * (A - 0.5)
    X2 = np.random.normal(0, sigma, n_samples) + 0.25 * (A - 0.5)

    X3 = np.random.normal(0, sigma, n_samples)
    X4 = np.random.normal(0, sigma, n_samples)

    Y = 2 * X1 + 4 * X2 + 1 * X3 + 1 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_model_subsets_design_6(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]

    X1 = np.random.normal(0, sigma, n_samples) + 0.5 * (A - 0.5)
    X3 = np.random.normal(0, sigma, n_samples)
    X2 = np.random.normal(0, sigma, n_samples) + 0.25 * (A - 0.5) + X3


    X4 = np.random.normal(0, sigma, n_samples)

    Y = 2 * X1 + 4 * X2 + 1 * X3 + 1 * X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_generate_data_models8_YindA(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, parameters[1], n_samples)
    sigma = parameters[2]
    X1 = 2*(2 * A - 1) + np.random.normal(0, sigma, n_samples)
    X2 = 2*(2 * A - 1) + np.random.normal(0, sigma, n_samples)

    X3 =  np.random.normal(0, sigma, n_samples)
    X4 =  np.random.normal(0, sigma, n_samples)

    Y = X1 + X2 + X3 + X4 + np.random.normal(0, sigma, n_samples) + (2 * A - 1)*parameters[0]
    Y = Y / np.std(Y)
    Y = np.where(Y > parameters[3], 1, 0)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    return df, features, sensitive_attribute, output, bins


def model_syn_single_f(alpha = 0.5):
    n_samples = 1000000
    np.random.seed(0)  # For reproducibility
    U = np.random.binomial(1, 0.75, n_samples)
    A = np.random.binomial(1, 0.5, n_samples)
    X = A * U
    Uy = np.random.binomial(1, alpha, n_samples)
    Y = Uy * A + (1 - U) * (1 - A)
    Uy = np.random.binomial(1, alpha, n_samples)
    Y = Uy * A + (1 - U) * (1 - A)

    # Compute m for P(hat_Y=1 | X=0)
    m = (1 / (1 - 0.5 * 0.75)) * (alpha * 0.5 * 0.25 + (1 - alpha) * 0.5 * 0.75)

    # Generate hat_Y
    hat_Y = np.zeros(n_samples)
    mask_x1 = (X == 1)
    mask_x0 = (X == 0)

    hat_Y[mask_x1] = 1
    hat_Y[mask_x0] = np.random.binomial(1, m, np.sum(mask_x0))
    df = pd.DataFrame({
        'X': X,
        'hat_Y': hat_Y,
        'A': A,
        'Y': Y
    })
    features = ['X', 'hat_Y']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (2,2,2,2)
    return df, features, sensitive_attribute, output, bins



def unified_model(Adj_M, Adj_N, pa, thy, n_samples=1000000, parameters=[], seed=0):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, 1, n_samples)
    u1 = np.random.normal(0, 1, n_samples)
    u2 = np.random.normal(0, 1, n_samples)
    u3 = np.random.normal(0, 1, n_samples)
    u4 = np.random.normal(0, 1, n_samples)
    a = np.random.binomial(1, pa, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution
    Z1 = np.random.uniform(0, 2, n_samples)
    Z2 = np.random.gamma(2, 1, n_samples)

    # Identity matrix
    I_6 = np.eye(6)

    # Construct U and Z
    U = np.column_stack([uy, u1, u2, u3, u4,a])  # Shape: (n_samples, 7)
    Z = np.column_stack([Z1, Z2])  # Shape: (n_samples, 3)

    # Compute V
    I_minus_Adj_M = I_6 - Adj_M
    V = np.linalg.inv(I_minus_Adj_M) @ U.T + np.linalg.inv(I_minus_Adj_M) @ Adj_N @ Z.T  # Shape: (7, n_samples)

    # Extract variables
    Y, X1, X2, X3, X4, A = V
    Y = (Y-np.mean(Y))/np.std(Y)
    Y = np.where(Y > thy, 1, 0)
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=False, N=5)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1 :
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0]-min(Dkl[1:]))/Dkl[0]<0.2 or (Dkl_y[0]-min(Dkl_y[1:]))/Dkl_y[0]<0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins

def unified_model_select_th(Adj_M, Adj_N, pa, thy, n_samples=1000000, parameters=[], seed=0):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, 1, n_samples)
    u1 = np.random.normal(0, 1, n_samples)
    u2 = np.random.normal(0, 1, n_samples)
    u3 = np.random.normal(0, 1, n_samples)
    u4 = np.random.normal(0, 1, n_samples)
    a = np.random.binomial(1, pa, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution
    Z1 = np.random.uniform(0, 2, n_samples)
    Z2 = np.random.gamma(2, 1, n_samples)

    # Identity matrix
    I_6 = np.eye(6)

    # Construct U and Z
    U = np.column_stack([uy, u1, u2, u3, u4,a])  # Shape: (n_samples, 7)
    Z = np.column_stack([Z1, Z2])  # Shape: (n_samples, 3)

    # Compute V
    I_minus_Adj_M = I_6 - Adj_M
    V = np.linalg.inv(I_minus_Adj_M) @ U.T + np.linalg.inv(I_minus_Adj_M) @ Adj_N @ Z.T  # Shape: (7, n_samples)

    # Extract variables
    Y, X1, X2, X3, X4, A = V
    Y = (Y-np.mean(Y))/np.std(Y)

    thy = -0.5
    max_diff = -1
    for th in np.arange(-0.5, 0.5, 0.1):
        Y1 = np.where(Y > th, 1, 0)
        df = pd.DataFrame({
            'X1': X1 / np.std(X1),
            'X2': X2 / np.std(X2),
            'X3': X3 / np.std(X3),
            'X4': X4 / np.std(X4),
            'A': A,
            'Y': Y1
        })
        features = ['X1', 'X2', 'X3', 'X4']
        sensitive_attribute = ['A']
        output = ['Y']
        for feature in features:
            df[feature] = pd.cut(df[feature], bins=6, labels=False)
        bins = (6, 6, 6, 6, 2, 2)
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=False, N=5)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1:
            # print("Model is almost fair.")
            continue
        if (Dkl[0] - min(Dkl[1:])) / Dkl[0] < 0.1 or (Dkl_y[0] - min(Dkl_y[1:])) / Dkl_y[0] < 0.1:
            continue
        var1 = np.maximum((Dkl[0] - Dkl[1:]) , 0)
        var2 = np.maximum((Dkl_y[0] - Dkl_y[1:]) , 0)
        var1= var1/np.sum(var1)
        var2= var2/np.sum(var2)
        diff = np.sum(np.abs((var1 - var2)))
        if diff > max_diff:
            max_diff = diff
            thy = th

    Y1 = np.where(Y > thy, 1, 0)
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y1
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)
    # print("max_diff: ", max_diff)
    if max_diff < 0.75:
        return df, [1], sensitive_attribute, output, bins, 0

    return df, features, sensitive_attribute, output, bins, thy

def create_adj_MN(pa, pd, thy, Nc):
    np.random.seed(int(time.time()))

    Adj_M = np.zeros((6, 6))
    # fill the upper diagonal matrix with zero or one randomly
    for i in range(6):
        for j in range(i + 1, 6):
            Adj_M[i, j] = np.random.binomial(1, pd, 1)

    # check that at least one exist for the first element
    if np.sum(Adj_M[0,1:5]) == 0:
        Adj_M[0, np.random.randint(1, 5)] = 1
    if np.sum(Adj_M[1:5, 5]) == 0:
        Adj_M[np.random.randint(1, 5), 5] = 1

    # Create matrix Adj_N to make Z1, Z2, Z3 confounders for X1, X2, X3, X4, X5: These correspond to the first 1 to 5 rows of Adj_N
    Adj_N = np.zeros((6, 2))
    k = [0, 0]
    if Nc == 1:
        k[0] = 1
    elif Nc == 2:
        k[0] = 1
        k[1] = 1

    for j in range(2):
        if k[j] == 0:
            continue
        for i in range(1,5):
            Adj_N[i, j] = np.random.binomial(1, pd, 1)
        while (np.sum(Adj_N[1:5, j]) < 2):
            rows = np.random.choice(range(1, 5), 1, replace=False)
            Adj_N[rows[0], j] = 1


    # each element in Adj_M and Adj_N is set to a value between -1 and 1 randomly if it is equal to 1
    for i in range(6):
        for j in range(6):
            if Adj_M[i, j] == 1:
                Adj_M[i, j] = np.random.uniform(-1, 1, 1)
    for i in range(6):
        for j in range(2):
            if Adj_N[i, j] == 1:
                Adj_N[i, j] = np.random.uniform(-1, 1, 1)


    return Adj_M, Adj_N, pa, thy

def create_adj_MN_new(pa, pd, thy, Nc):
    np.random.seed(int(time.time()))

    Adj_M = np.zeros((6, 6))
    # fill the upper diagonal matrix with zero or one randomly
    for i in range(6):
        for j in range(i + 1, 6):
            Adj_M[i, j] = np.random.binomial(1, pd, 1)


    # check that at least one exist for the first element
    if np.sum(Adj_M[0,1:5]) == 0:
        Adj_M[0, np.random.randint(1, 5)] = 1
    if np.sum(Adj_M[1:5, 5]) == 0:
        Adj_M[np.random.randint(1, 5), 5] = 1
    Adj_M[0, 5] = np.random.binomial(1, 0.9, 1)
    # Create matrix Adj_N to make Z1, Z2, Z3 confounders for X1, X2, X3, X4, X5: These correspond to the first 1 to 5 rows of Adj_N
    Adj_N = np.zeros((6, 2))
    k = [0, 0]
    if Nc == 1:
        k[1] = 1
    elif Nc == 2:
        k[0] = 1
        k[1] = 1

    for j in range(2):
        if k[j] == 0:
            continue
        for i in range(1,5):
            Adj_N[i, j] = np.random.binomial(1, pd, 1)
        while (np.sum(Adj_N[1:5, j]) < 2):
            rows = np.random.choice(range(1, 5), 1, replace=False)
            Adj_N[rows[0], j] = 1


    # each element in Adj_M and Adj_N is set to a value between -1 and 1 randomly if it is equal to 1
    for i in range(6):
        for j in range(6):
            if Adj_M[i, j] == 1:
                v1 = np.random.uniform(-1, -0.5, 1)
                v2 = np.random.uniform(0.5, 1, 1)
                p = np.random.binomial(1, 0.75, 1)
                if p == 1:
                    Adj_M[i, j] = v2
                else:
                    Adj_M[i, j] = v1
                # Adj_M[i, j] = np.random.choice([v1, v2], 1)
                # Adj_M[i, j] = np.random.uniform(-1, 1, 1)
    for i in range(6):
        for j in range(2):
            if Adj_N[i, j] == 1:
                v1 = np.random.uniform(-1, -0.5, 1)
                v2 = np.random.uniform(0.5, 1, 1)
                p = np.random.binomial(1, 0.75, 1)
                if p == 1:
                    Adj_N[i, j] = v2
                else:
                    Adj_N[i, j] = v1
                # Adj_N[i, j] = np.random.uniform(-1, 1, 1)


    return Adj_M, Adj_N, pa, thy

def create_adj_MN_new_positive_only(pa, pd, thy, Nc):
    np.random.seed(int(time.time()))

    Adj_M = np.zeros((6, 6))
    # fill the upper diagonal matrix with zero or one randomly
    for i in range(6):
        for j in range(i + 1, 6):
            Adj_M[i, j] = np.random.binomial(1, pd, 1)


    # check that at least one exist for the first element
    if np.sum(Adj_M[0,1:5]) == 0:
        Adj_M[0, np.random.randint(1, 5)] = 1
    if np.sum(Adj_M[1:5, 5]) == 0:
        Adj_M[np.random.randint(1, 5), 5] = 1
    Adj_M[0, 5] = np.random.binomial(1, 0.9, 1)
    # Create matrix Adj_N to make Z1, Z2, Z3 confounders for X1, X2, X3, X4, X5: These correspond to the first 1 to 5 rows of Adj_N
    Adj_N = np.zeros((6, 2))
    k = [0, 0]
    if Nc == 1:
        k[1] = 1
    elif Nc == 2:
        k[0] = 1
        k[1] = 1

    for j in range(2):
        if k[j] == 0:
            continue
        for i in range(1,5):
            Adj_N[i, j] = np.random.binomial(1, pd, 1)
        while (np.sum(Adj_N[1:5, j]) < 2):
            rows = np.random.choice(range(1, 5), 1, replace=False)
            Adj_N[rows[0], j] = 1


    # each element in Adj_M and Adj_N is set to a value between -1 and 1 randomly if it is equal to 1
    for i in range(6):
        for j in range(6):
            if Adj_M[i, j] == 1:
                v1 = np.random.uniform(-1, -0.5, 1)
                v2 = np.random.uniform(0.5, 1.5, 1)
                p = np.random.binomial(1, 0.75, 1)
                Adj_M[i, j] = v2
                # Adj_M[i, j] = np.random.choice([v1, v2], 1)
                # Adj_M[i, j] = np.random.uniform(-1, 1, 1)
    for i in range(6):
        for j in range(2):
            if Adj_N[i, j] == 1:
                v1 = np.random.uniform(-1, -0.5, 1)
                v2 = np.random.uniform(0.5, 1.5, 1)
                p = np.random.binomial(1, 0.75, 1)
                Adj_N[i, j] = v2
                # Adj_N[i, j] = np.random.uniform(-1, 1, 1)


    return Adj_M, Adj_N, pa, thy


def create_adj_MN_testing(pa, pd, thy, Nc):
    np.random.seed(int(time.time()))

    Adj_M = np.zeros((6, 6))
    # fill the upper diagonal matrix with zero or one randomly
    for i in range(6):
        for j in range(i + 1, 6):
            Adj_M[i, j] = np.random.binomial(1, pd, 1)


    # check that at least one exist for the first element
    if np.sum(Adj_M[0,1:5]) == 0:
        Adj_M[0, np.random.randint(1, 5)] = 1
    if np.sum(Adj_M[1:5, 5]) == 0:
        Adj_M[np.random.randint(1, 5), 5] = 1

    Adj_M[0, 5] = np.random.binomial(1, 0.9, 1)
    # Create matrix Adj_N to make Z1, Z2, Z3 confounders for X1, X2, X3, X4, X5: These correspond to the first 1 to 5 rows of Adj_N
    Adj_N = np.zeros((6, 2))
    k = [0, 0]
    if Nc == 1:
        k[1] = 1
    elif Nc == 2:
        k[0] = 1
        k[1] = 1

    for j in range(2):
        if k[j] == 0:
            continue
        for i in range(1,5):
            Adj_N[i, j] = np.random.binomial(1, pd, 1)
        while (np.sum(Adj_N[1:5, j]) < 2):
            rows = np.random.choice(range(1, 5), 1, replace=False)
            Adj_N[rows[0], j] = 1


    # each element in Adj_M and Adj_N is set to a value between -1 and 1 randomly if it is equal to 1
    for i in range(6):
        for j in range(6):
            if Adj_M[i, j] == 1:
                v1 = np.random.uniform(-1, -0.5, 1)
                v2 = np.random.uniform(1, 2, 1)
                p = np.random.binomial(1, 0.75, 1)
                Adj_M[i, j] = v2
                # Adj_M[i, j] = np.random.choice([v1, v2], 1)
                # Adj_M[i, j] = np.random.uniform(-1, 1, 1)
    for i in range(6):
        for j in range(2):
            if Adj_N[i, j] == 1:
                v1 = np.random.uniform(-1, -0.5, 1)
                v2 = np.random.uniform(1, 2, 1)
                p = np.random.binomial(1, 0.75, 1)
                Adj_N[i, j] = v2
                # Adj_N[i, j] = np.random.uniform(-1, 1, 1)
    if Adj_M[0, 5] != 0:
        Adj_M[0, 5] = 2

    return Adj_M, Adj_N, pa, thy


def create_adj_MN_verified(pa, pdx, pdy, pday, thy, Nc, max_w = 2, min_w = 1):
    np.random.seed(int(time.time()))
    # print("int(time.time())",int(time.time()))

    Adj_M = np.zeros((6, 6))
    # fill the upper diagonal matrix with zero or one randomly
    for i in range(6):
        for j in range(i + 1, 6):
            Adj_M[i, j] = np.random.binomial(1, pdx, 1)

    for j in range(1,6):
        Adj_M[0, j] = np.random.binomial(1, pdy, 1)

    Adj_M[0, 5] = np.random.binomial(1, pday, 1)
    # Create matrix Adj_N to make Z1, Z2, Z3 confounders for X1, X2, X3, X4, X5: These correspond to the first 1 to 5 rows of Adj_N
    Adj_N = np.zeros((6, 2))
    k = [0, 0]
    if Nc == 1:
        k[0] = 1
    elif Nc == 2:
        k[0] = 1
        k[1] = 1

    for j in range(2):
        if k[j] == 0:
            continue
        for i in range(0, 5):
            Adj_N[i, j] = np.random.binomial(1, pdx, 1)
        while (np.sum(Adj_N[0:5, j]) < 2):
            rows = np.random.choice(range(0, 5), 1, replace=False)
            Adj_N[rows[0], j] = 1

    # print("Adj_M")
    # print(Adj_M)
    # print("Adj_N")
    # print(Adj_N)
    if Nc == 0:
        M = Adj_M
        # print("M")
        # print(M)
        M = M + M.T
        # check if M is adjacent to a connected graph
    if Nc == 1:
        M = np.zeros((7, 7))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6] = Adj_N[:, 0]
        # print("M")
        # print(M)
        M = M + M.T
    if Nc == 2:
        M = np.zeros((8, 8))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6:8] = Adj_N
        # print("M")
        # print(M)
        M = M + M.T

    # print("M")
    # print(M)

    n_components = connected_components(csgraph=M, directed=False, return_labels=False)
    # print("n_components", n_components)
    if n_components > 1:
        return np.array([]), Adj_N, pa, thy

    # each element in Adj_M and Adj_N is set to a value between -1 and 1 randomly if it is equal to 1
    for i in range(6):
        for j in range(6):
            if Adj_M[i, j] == 1:
                v1 = np.random.uniform(-1, -0.5, 1)
                v2 = np.random.uniform(min_w, max_w, 1)
                p = np.random.binomial(1, 0.75, 1)
                Adj_M[i, j] = v2
                # Adj_M[i, j] = np.random.choice([v1, v2], 1)
                # Adj_M[i, j] = np.random.uniform(-1, 1, 1)
    for i in range(6):
        for j in range(2):
            if Adj_N[i, j] == 1:
                v1 = np.random.uniform(-1, -0.5, 1)
                v2 = np.random.uniform(min_w, max_w, 1)
                p = np.random.binomial(1, 0.75, 1)
                Adj_N[i, j] = v2
                # Adj_N[i, j] = np.random.uniform(-1, 1, 1)
    if Adj_M[0, 5] != 0:
        Adj_M[0, 5] = max_w

    return Adj_M, Adj_N, pa, thy


def create_adj_MN_verified_updated(p_a, pd_ax, pd_x_to_y, N_a_to_y, Nc, thy, max_w = 2, min_w = 1, p_c_to_y = 0.5):
    np.random.seed(time.time_ns() % (2**32))
    # print("int(time.time())",(time.time_ns() % (2**32)))

    Adj_M = np.zeros((6, 6))
    # fill the upper diagonal matrix with zero or one randomly
    for i in range(6):
        for j in range(i + 1, 6):
            Adj_M[i, j] = np.random.binomial(1, pd_ax, 1)

    for j in range(1,6):
        Adj_M[0, j] = np.random.binomial(1, pd_x_to_y, 1)

    Adj_M[0, 5] = N_a_to_y
    # Create matrix Adj_N to make Z1, Z2, Z3 confounders for X1, X2, X3, X4, X5: These correspond to the first 1 to 5 rows of Adj_N
    Adj_N = np.zeros((6, 2))
    k = [0, 0]
    if Nc == 1:
        k[0] = 1
    elif Nc == 2:
        k[0] = 1
        k[1] = 1

    for j in range(2):
        if k[j] == 0:
            continue
        for i in range(1, 5):
            Adj_N[i, j] = np.random.binomial(1, pd_ax, 1)
        Adj_N[0, j] = np.random.binomial(1, p_c_to_y, 1)
        while (np.sum(Adj_N[0:5, j]) < 2):
            rows = np.random.choice(range(0, 5), 1, replace=False)
            Adj_N[rows[0], j] = 1

    # print("Adj_M")
    # print(Adj_M)
    # print("Adj_N")
    # print(Adj_N)
    if Nc == 0:
        M = Adj_M
        # print("M")
        # print(M)
        M = M + M.T
        # check if M is adjacent to a connected graph
    if Nc == 1:
        M = np.zeros((7, 7))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6] = Adj_N[:, 0]
        # print("M")
        # print(M)
        M = M + M.T
    if Nc == 2:
        M = np.zeros((8, 8))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6:8] = Adj_N
        # print("M")
        # print(M)
        M = M + M.T

    # print("M")
    # print(M)

    n_components = connected_components(csgraph=M, directed=False, return_labels=False)
    # print("n_components", n_components)
    if n_components > 1:
        return np.array([]), Adj_N, p_a, thy

    # each element in Adj_M and Adj_N is set to a value between -1 and 1 randomly if it is equal to 1
    for i in range(6):
        for j in range(6):
            if Adj_M[i, j] == 1:
                Adj_M[i, j] = np.random.uniform(min_w, max_w, 1)

    for i in range(6):
        for j in range(2):
            if Adj_N[i, j] == 1:
                Adj_N[i, j] = np.random.uniform(min_w, max_w, 1)
    if Adj_M[0, 5] != 0:
        Adj_M[0, 5] = max_w

    return Adj_M, Adj_N, p_a, thy

def create_adj_MN_verified_updated_pos_neg(p_a, pd_ax, pd_x_to_y, N_a_to_y, Nc, thy, max_w = 2, min_w = 1, p_c_to_y = 0.5):
    np.random.seed(time.time_ns() % (2**32))
    # print("int(time.time())",(time.time_ns() % (2**32)))

    Adj_M = np.zeros((6, 6))
    # fill the upper diagonal matrix with zero or one randomly
    for i in range(6):
        for j in range(i + 1, 6):
            Adj_M[i, j] = np.random.binomial(1, pd_ax, 1)

    for j in range(1,6):
        Adj_M[0, j] = np.random.binomial(1, pd_x_to_y, 1)

    Adj_M[0, 5] = N_a_to_y
    # Create matrix Adj_N to make Z1, Z2, Z3 confounders for X1, X2, X3, X4, X5: These correspond to the first 1 to 5 rows of Adj_N
    Adj_N = np.zeros((6, 2))
    k = [0, 0]
    if Nc == 1:
        k[0] = 1
    elif Nc == 2:
        k[0] = 1
        k[1] = 1

    for j in range(2):
        if k[j] == 0:
            continue
        for i in range(1, 5):
            Adj_N[i, j] = np.random.binomial(1, pd_ax, 1)
        Adj_N[0, j] = np.random.binomial(1, p_c_to_y, 1)
        while (np.sum(Adj_N[0:5, j]) < 2):
            rows = np.random.choice(range(0, 5), 1, replace=False)
            Adj_N[rows[0], j] = 1

    # print("Adj_M")
    # print(Adj_M)
    # print("Adj_N")
    # print(Adj_N)
    if Nc == 0:
        M = Adj_M
        # print("M")
        # print(M)
        M = M + M.T
        # check if M is adjacent to a connected graph
    if Nc == 1:
        M = np.zeros((7, 7))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6] = Adj_N[:, 0]
        # print("M")
        # print(M)
        M = M + M.T
    if Nc == 2:
        M = np.zeros((8, 8))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6:8] = Adj_N
        # print("M")
        # print(M)
        M = M + M.T

    # print("M")
    # print(M)

    n_components = connected_components(csgraph=M, directed=False, return_labels=False)
    # print("n_components", n_components)
    if n_components > 1:
        return np.array([]), Adj_N, p_a, thy

    # each element in Adj_M and Adj_N is set to a value between -1 and 1 randomly if it is equal to 1
    for i in range(6):
        for j in range(6):
            if Adj_M[i, j] == 1:
                g1 = np.random.uniform(-max_w, -min_w, 1)
                g2 = np.random.uniform(min_w, max_w, 1)
                select_var = np.random.binomial(1, 0.5, 1)
                Adj_M[i, j] = g1 if select_var == 0 else g2
                # Adj_M[i, j] = np.random.choice([g1, g2], 1)
                # print("Adj_M[i, j]", Adj_M[i, j])

                # Adj_M[i, j] = np.random.uniform(min_w, max_w, 1)

    for i in range(6):
        for j in range(2):
            if Adj_N[i, j] == 1:
                g1 = np.random.uniform(-max_w, -min_w, 1)
                g2 = np.random.uniform(min_w, max_w, 1)
                select_var = np.random.binomial(1, 0.5, 1)
                Adj_N[i, j] = g1 if select_var == 0 else g2

                # Adj_N[i, j] = np.random.uniform(min_w, max_w, 1)
    if Adj_M[0, 5] != 0:
        Adj_M[0, 5] = max_w

    return Adj_M, Adj_N, p_a, thy

def create_adj_MN_verified_updated_pos_neg_pa_dis(p_a, pd_ax, pd_x_to_y, N_a_to_y, Nc, thy, max_w = 2, min_w = 1, p_c_to_y = 0.5, dis_pa= 0.75):
    np.random.seed(time.time_ns() % (2**32))
    # print("int(time.time())",(time.time_ns() % (2**32)))

    Adj_M = np.zeros((6, 6))
    # fill the upper diagonal matrix with zero or one randomly
    for i in range(6):
        for j in range(i + 1, 6):
            Adj_M[i, j] = np.random.binomial(1, pd_ax, 1)

    for j in range(1,6):
        Adj_M[0, j] = np.random.binomial(1, pd_x_to_y, 1)

    Adj_M[0, 5] = N_a_to_y*np.random.binomial(1, dis_pa, 1)
    # Create matrix Adj_N to make Z1, Z2, Z3 confounders for X1, X2, X3, X4, X5: These correspond to the first 1 to 5 rows of Adj_N
    Adj_N = np.zeros((6, 2))
    k = [0, 0]
    if Nc == 1:
        k[0] = 1
    elif Nc == 2:
        k[0] = 1
        k[1] = 1

    for j in range(2):
        if k[j] == 0:
            continue
        for i in range(1, 5):
            Adj_N[i, j] = np.random.binomial(1, pd_ax, 1)
        Adj_N[0, j] = np.random.binomial(1, p_c_to_y, 1)
        while (np.sum(Adj_N[0:5, j]) < 2):
            rows = np.random.choice(range(0, 5), 1, replace=False)
            Adj_N[rows[0], j] = 1

    # print("Adj_M")
    # print(Adj_M)
    # print("Adj_N")
    # print(Adj_N)
    if Nc == 0:
        M = Adj_M
        # print("M")
        # print(M)
        M = M + M.T
        # check if M is adjacent to a connected graph
    if Nc == 1:
        M = np.zeros((7, 7))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6] = Adj_N[:, 0]
        # print("M")
        # print(M)
        M = M + M.T
    if Nc == 2:
        M = np.zeros((8, 8))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6:8] = Adj_N
        # print("M")
        # print(M)
        M = M + M.T

    # print("M")
    # print(M)

    n_components = connected_components(csgraph=M, directed=False, return_labels=False)
    # print("n_components", n_components)
    if n_components > 1:
        return np.array([]), Adj_N, p_a, thy

    # each element in Adj_M and Adj_N is set to a value between -1 and 1 randomly if it is equal to 1
    for i in range(6):
        for j in range(6):
            if Adj_M[i, j] == 1:
                g1 = np.random.uniform(-max_w, -min_w, 1)
                g2 = np.random.uniform(min_w, max_w, 1)
                select_var = np.random.binomial(1, 0.5, 1)
                Adj_M[i, j] = g1 if select_var == 0 else g2
                # Adj_M[i, j] = np.random.choice([g1, g2], 1)
                # print("Adj_M[i, j]", Adj_M[i, j])

                # Adj_M[i, j] = np.random.uniform(min_w, max_w, 1)

    for i in range(6):
        for j in range(2):
            if Adj_N[i, j] == 1:
                g1 = np.random.uniform(-max_w, -min_w, 1)
                g2 = np.random.uniform(min_w, max_w, 1)
                select_var = np.random.binomial(1, 0.5, 1)
                Adj_N[i, j] = g1 if select_var == 0 else g2

                # Adj_N[i, j] = np.random.uniform(min_w, max_w, 1)
    if Adj_M[0, 5] != 0:
        Adj_M[0, 5] = max_w

    return Adj_M, Adj_N, p_a, thy

def create_adj_MN_verified_updated_pos_neg_pa_pos_neg(p_a, pd_ax, pd_x_to_y, N_a_to_y, Nc, thy, max_w = 2, min_w = 1, p_c_to_y = 0.5, dis_pa= 0.75, pos_neg_a = False):
    np.random.seed(time.time_ns() % (2**32))
    # print("int(time.time())",(time.time_ns() % (2**32)))

    Adj_M = np.zeros((6, 6))
    # fill the upper diagonal matrix with zero or one randomly
    for i in range(6):
        for j in range(i + 1, 6):
            Adj_M[i, j] = np.random.binomial(1, pd_ax, 1)

    for j in range(1,6):
        Adj_M[0, j] = np.random.binomial(1, pd_x_to_y, 1)

    Adj_M[0, 5] = N_a_to_y*np.random.binomial(1, dis_pa, 1)
    # Create matrix Adj_N to make Z1, Z2, Z3 confounders for X1, X2, X3, X4, X5: These correspond to the first 1 to 5 rows of Adj_N
    Adj_N = np.zeros((6, 2))
    k = [0, 0]
    if Nc == 1:
        k[0] = 1
    elif Nc == 2:
        k[0] = 1
        k[1] = 1

    for j in range(2):
        if k[j] == 0:
            continue
        for i in range(1, 5):
            Adj_N[i, j] = np.random.binomial(1, pd_ax, 1)
        Adj_N[0, j] = np.random.binomial(1, p_c_to_y, 1)
        while (np.sum(Adj_N[0:5, j]) < 2):
            rows = np.random.choice(range(0, 5), 1, replace=False)
            Adj_N[rows[0], j] = 1

    # print("Adj_M")
    # print(Adj_M)
    # print("Adj_N")
    # print(Adj_N)
    if Nc == 0:
        M = Adj_M
        # print("M")
        # print(M)
        M = M + M.T
        # check if M is adjacent to a connected graph
    if Nc == 1:
        M = np.zeros((7, 7))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6] = Adj_N[:, 0]
        # print("M")
        # print(M)
        M = M + M.T
    if Nc == 2:
        M = np.zeros((8, 8))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6:8] = Adj_N
        # print("M")
        # print(M)
        M = M + M.T

    # print("M")
    # print(M)

    n_components = connected_components(csgraph=M, directed=False, return_labels=False)
    # print("n_components", n_components)
    if n_components > 1:
        return np.array([]), Adj_N, p_a, thy

    # each element in Adj_M and Adj_N is set to a value between -1 and 1 randomly if it is equal to 1
    for i in range(6):
        for j in range(6):
            if Adj_M[i, j] == 1:
                g1 = np.random.uniform(-max_w, -min_w, 1)
                g2 = np.random.uniform(min_w, max_w, 1)
                select_var = np.random.binomial(1, 0.5, 1)
                Adj_M[i, j] = g1 if select_var == 0 else g2
                # Adj_M[i, j] = np.random.choice([g1, g2], 1)
                # print("Adj_M[i, j]", Adj_M[i, j])

                # Adj_M[i, j] = np.random.uniform(min_w, max_w, 1)

    for i in range(6):
        for j in range(2):
            if Adj_N[i, j] == 1:
                g1 = np.random.uniform(-max_w, -min_w, 1)
                g2 = np.random.uniform(min_w, max_w, 1)
                select_var = np.random.binomial(1, 0.5, 1)
                Adj_N[i, j] = g1 if select_var == 0 else g2

                # Adj_N[i, j] = np.random.uniform(min_w, max_w, 1)
    if Adj_M[0, 5] != 0:
        g1 = np.random.uniform(-max_w, -min_w, 1)
        g2 = np.random.uniform(min_w, max_w, 1)
        select_var = np.random.binomial(1, 0.5, 1)
        if pos_neg_a == True:
            Adj_M[0, 5] = g1 if select_var == 0 else g2
        else:
            Adj_M[0, 5] = g2

    return Adj_M, Adj_N, p_a, thy


def create_adj_MN_verified_updated_pos_neg_pa_pos_neg_bern(p_a, pd_ax, pd_x_to_y, N_a_to_y, Nc, thy, max_w = 2, min_w = 1, p_c_to_y = 0.5, pos_neg_a = False):
    np.random.seed(time.time_ns() % (2**32))
    # print("int(time.time())",(time.time_ns() % (2**32)))

    Adj_M = np.zeros((6, 6))
    # fill the upper diagonal matrix with zero or one randomly
    for i in range(6):
        for j in range(i + 1, 6):
            Adj_M[i, j] = np.random.binomial(1, pd_ax, 1)

    for j in range(1,6):
        Adj_M[0, j] = np.random.binomial(1, pd_x_to_y, 1)

    Adj_M[0, 5] = np.random.binomial(1, N_a_to_y, 1)
    # Create matrix Adj_N to make Z1, Z2, Z3 confounders for X1, X2, X3, X4, X5: These correspond to the first 1 to 5 rows of Adj_N
    Adj_N = np.zeros((6, 2))
    k = [0, 0]
    if Nc == 1:
        k[0] = 1
    elif Nc == 2:
        k[0] = 1
        k[1] = 1

    for j in range(2):
        if k[j] == 0:
            continue
        for i in range(1, 5):
            Adj_N[i, j] = np.random.binomial(1, pd_ax, 1)
        Adj_N[0, j] = np.random.binomial(1, p_c_to_y, 1)
        while (np.sum(Adj_N[0:5, j]) < 2):
            rows = np.random.choice(range(0, 5), 1, replace=False)
            Adj_N[rows[0], j] = 1

    # print("Adj_M")
    # print(Adj_M)
    # print("Adj_N")
    # print(Adj_N)
    if Nc == 0:
        M = Adj_M
        # print("M")
        # print(M)
        M = M + M.T
        # check if M is adjacent to a connected graph
    if Nc == 1:
        M = np.zeros((7, 7))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6] = Adj_N[:, 0]
        # print("M")
        # print(M)
        M = M + M.T
    if Nc == 2:
        M = np.zeros((8, 8))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6:8] = Adj_N
        # print("M")
        # print(M)
        M = M + M.T

    # print("M")
    # print(M)

    n_components = connected_components(csgraph=M, directed=False, return_labels=False)
    # print("n_components", n_components)
    if n_components > 1:
        return np.array([]), Adj_N, p_a, thy

    # each element in Adj_M and Adj_N is set to a value between -1 and 1 randomly if it is equal to 1
    for i in range(6):
        for j in range(6):
            if Adj_M[i, j] == 1:
                g1 = np.random.uniform(-max_w, -min_w, 1)
                g2 = np.random.uniform(min_w, max_w, 1)
                select_var = np.random.binomial(1, 0.5, 1)
                Adj_M[i, j] = g1 if select_var == 0 else g2
                # Adj_M[i, j] = np.random.choice([g1, g2], 1)
                # print("Adj_M[i, j]", Adj_M[i, j])

                # Adj_M[i, j] = np.random.uniform(min_w, max_w, 1)

    for i in range(6):
        for j in range(2):
            if Adj_N[i, j] == 1:
                g1 = np.random.uniform(-max_w, -min_w, 1)
                g2 = np.random.uniform(min_w, max_w, 1)
                select_var = np.random.binomial(1, 0.5, 1)
                Adj_N[i, j] = g1 if select_var == 0 else g2

                # Adj_N[i, j] = np.random.uniform(min_w, max_w, 1)
    if Adj_M[0, 5] != 0:
        g1 = np.random.uniform(-max_w, -min_w, 1)
        g2 = np.random.uniform(min_w, max_w, 1)
        select_var = np.random.binomial(1, 0.5, 1)
        if pos_neg_a == True:
            Adj_M[0, 5] = g1 if select_var == 0 else g2
        else:
            Adj_M[0, 5] = g2

    return Adj_M, Adj_N, p_a, thy


def create_adj_MN_verified_updated_pos_neg_pa_pos_neg_bern_3con(p_a, pd_ax, pd_x_to_y, N_a_to_y, Nc, thy, max_w = 2, min_w = 1, p_c_to_y = 0.5, pos_neg_a = False):
    np.random.seed(time.time_ns() % (2**32))
    # print("int(time.time())",(time.time_ns() % (2**32)))

    Adj_M = np.zeros((6, 6))
    # fill the upper diagonal matrix with zero or one randomly
    for i in range(6):
        for j in range(i + 1, 6):
            Adj_M[i, j] = np.random.binomial(1, pd_ax, 1)

    for j in range(1,6):
        Adj_M[0, j] = np.random.binomial(1, pd_x_to_y, 1)

    Adj_M[0, 5] = np.random.binomial(1, N_a_to_y, 1)
    # Create matrix Adj_N to make Z1, Z2, Z3 confounders for X1, X2, X3, X4, X5: These correspond to the first 1 to 5 rows of Adj_N
    Adj_N = np.zeros((6, 3))
    k = [0, 0, 0]
    if Nc == 1:
        k[0] = 1
    elif Nc == 2:
        k[0] = 1
        k[1] = 1
    elif Nc == 3:
        k[0] = 1
        k[1] = 1
        k[2] = 1

    for j in range(3):
        if k[j] == 0:
            continue
        for i in range(1, 5):
            Adj_N[i, j] = np.random.binomial(1, pd_ax, 1)
        Adj_N[0, j] = np.random.binomial(1, p_c_to_y, 1)
        while (np.sum(Adj_N[0:5, j]) < 2):
            rows = np.random.choice(range(0, 5), 1, replace=False)
            Adj_N[rows[0], j] = 1

    # print("Adj_M")
    # print(Adj_M)
    # print("Adj_N")
    # print(Adj_N)
    if Nc == 0:
        M = Adj_M
        # print("M")
        # print(M)
        M = M + M.T
        # check if M is adjacent to a connected graph
    if Nc == 1:
        M = np.zeros((7, 7))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6] = Adj_N[:, 0]
        # print("M")
        # print(M)
        M = M + M.T
    if Nc == 2:
        M = np.zeros((8, 8))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6:8] = Adj_N[:, 0:2]
        # print("M")
        # print(M)
        M = M + M.T
    if Nc == 3:
        M = np.zeros((9, 9))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6:9] = Adj_N
        # print("M")
        # print(M)
        M = M + M.T

    # print("M")
    # print(M)

    n_components = connected_components(csgraph=M, directed=False, return_labels=False)
    # print("n_components", n_components)
    if n_components > 1:
        return np.array([]), Adj_N, p_a, thy

    # each element in Adj_M and Adj_N is set to a value between -1 and 1 randomly if it is equal to 1
    for i in range(6):
        for j in range(6):
            if Adj_M[i, j] == 1:
                g1 = np.random.uniform(-max_w, -min_w, 1)
                g2 = np.random.uniform(min_w, max_w, 1)
                select_var = np.random.binomial(1, 0.5, 1)
                Adj_M[i, j] = g1 if select_var == 0 else g2
                # Adj_M[i, j] = np.random.choice([g1, g2], 1)
                # print("Adj_M[i, j]", Adj_M[i, j])

                # Adj_M[i, j] = np.random.uniform(min_w, max_w, 1)

    for i in range(6):
        for j in range(3):
            if Adj_N[i, j] == 1:
                g1 = np.random.uniform(-max_w, -min_w, 1)
                g2 = np.random.uniform(min_w, max_w, 1)
                select_var = np.random.binomial(1, 0.5, 1)
                Adj_N[i, j] = g1 if select_var == 0 else g2

                # Adj_N[i, j] = np.random.uniform(min_w, max_w, 1)
    if Adj_M[0, 5] != 0:
        g1 = np.random.uniform(-max_w, -min_w, 1)
        g2 = np.random.uniform(min_w, max_w, 1)
        select_var = np.random.binomial(1, 0.5, 1)
        if pos_neg_a == True:
            Adj_M[0, 5] = g1 if select_var == 0 else g2
        else:
            Adj_M[0, 5] = g2

    return Adj_M, Adj_N, p_a, thy


def create_adj_MN_verified_updated_pos_neg_pa_pos_neg_bern_3con_Amax(p_a, pd_ax, pd_x_to_y, N_a_to_y, Nc, thy, max_w = 2, min_w = 1, p_c_to_y = 0.5):
    # np.random.seed(time.time_ns() % (2**32))
    # print("int(time.time())",(time.time_ns() % (2**32)))

    Adj_M = np.zeros((6, 6))
    # fill the upper diagonal matrix with zero or one randomly
    for i in range(6):
        for j in range(i + 1, 6):
            Adj_M[i, j] = np.random.binomial(1, pd_ax, 1)

    for j in range(1,6):
        Adj_M[0, j] = np.random.binomial(1, pd_x_to_y, 1)

    Adj_M[0, 5] = np.random.binomial(1, N_a_to_y, 1)
    # Create matrix Adj_N to make Z1, Z2, Z3 confounders for X1, X2, X3, X4, X5: These correspond to the first 1 to 5 rows of Adj_N
    Adj_N = np.zeros((6, 3))
    k = [0, 0, 0]
    if Nc == 1:
        k[0] = 1
    elif Nc == 2:
        k[0] = 1
        k[1] = 1
    elif Nc == 3:
        k[0] = 1
        k[1] = 1
        k[2] = 1

    for j in range(3):
        if k[j] == 0:
            continue
        for i in range(1, 5):
            Adj_N[i, j] = np.random.binomial(1, pd_ax, 1)
        Adj_N[0, j] = np.random.binomial(1, p_c_to_y, 1)
        while (np.sum(Adj_N[0:5, j]) < 2):
            rows = np.random.choice(range(0, 5), 1, replace=False)
            Adj_N[rows[0], j] = 1

    # print("Adj_M")
    # print(Adj_M)
    # print("Adj_N")
    # print(Adj_N)
    if Nc == 0:
        M = Adj_M
        # print("M")
        # print(M)
        M = M + M.T
        # check if M is adjacent to a connected graph
    if Nc == 1:
        M = np.zeros((7, 7))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6] = Adj_N[:, 0]
        # print("M")
        # print(M)
        M = M + M.T
    if Nc == 2:
        M = np.zeros((8, 8))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6:8] = Adj_N[:, 0:2]
        # print("M")
        # print(M)
        M = M + M.T
    if Nc == 3:
        M = np.zeros((9, 9))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6:9] = Adj_N
        # print("M")
        # print(M)
        M = M + M.T

    # print("M")
    # print(M)

    n_components = connected_components(csgraph=M, directed=False, return_labels=False)
    # print("n_components", n_components)
    if n_components > 1:
        return np.array([]), Adj_N, p_a, thy

    # each element in Adj_M and Adj_N is set to a value between -1 and 1 randomly if it is equal to 1
    for i in range(6):
        for j in range(6):
            if Adj_M[i, j] == 1:
                g2 = np.random.uniform(min_w, max_w, 1)
                Adj_M[i, j] = g2

    for i in range(6):
        for j in range(3):
            if Adj_N[i, j] == 1:
                g2 = np.random.uniform(min_w, max_w, 1)
                Adj_N[i, j] = g2

    if Adj_M[0, 5] != 0:
        Adj_M[0, 5] = max_w


    return Adj_M, Adj_N, p_a, thy



def create_adj_MN_verified_updated_test(p_a, pd_ax, pd_x_to_y, N_a_to_y, Nc, thy, max_w = 2, min_w = 1, p_c_to_y = 0.5):
    np.random.seed(time.time_ns() % (2**32))
    # print("int(time.time())",(time.time_ns() % (2**32)))

    Adj_M = np.zeros((6, 6))
    # fill the upper diagonal matrix with zero or one randomly
    for i in range(6):
        for j in range(i + 1, 6):
            Adj_M[i, j] = np.random.binomial(1, pd_ax, 1)

    for j in range(1,6):
        Adj_M[0, j] = np.random.binomial(1, pd_x_to_y, 1)

    Adj_M[0, 5] = N_a_to_y
    # Create matrix Adj_N to make Z1, Z2, Z3 confounders for X1, X2, X3, X4, X5: These correspond to the first 1 to 5 rows of Adj_N
    Adj_N = np.zeros((6, 2))
    k = [0, 0]
    if Nc == 1:
        k[0] = 1
    elif Nc == 2:
        k[0] = 1
        k[1] = 1

    for j in range(2):
        if k[j] == 0:
            continue
        for i in range(1, 5):
            Adj_N[i, j] = np.random.binomial(1, pd_ax, 1)
        Adj_N[0, j] = np.random.binomial(1, p_c_to_y, 1)
        while (np.sum(Adj_N[0:5, j]) < 2):
            rows = np.random.choice(range(0, 5), 1, replace=False)
            Adj_N[rows[0], j] = 1

    # print("Adj_M")
    # print(Adj_M)
    # print("Adj_N")
    # print(Adj_N)
    if Nc == 0:
        M = Adj_M
        # print("M")
        # print(M)
        M = M + M.T
        # check if M is adjacent to a connected graph
    if Nc == 1:
        M = np.zeros((7, 7))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6] = Adj_N[:, 0]
        # print("M")
        # print(M)
        M = M + M.T
    if Nc == 2:
        M = np.zeros((8, 8))
        M[0:6, 0:6] = Adj_M
        M[0:6, 6:8] = Adj_N
        # print("M")
        # print(M)
        M = M + M.T

    # print("M")
    # print(M)

    n_components = connected_components(csgraph=M, directed=False, return_labels=False)
    # print("n_components", n_components)
    if n_components > 1:
        return np.array([]), Adj_N, p_a, thy

    # each element in Adj_M and Adj_N is set to a value between -1 and 1 randomly if it is equal to 1
    for i in range(6):
        for j in range(6):
            if Adj_M[i, j] == 1:
                Adj_M[i, j] = np.random.uniform(min_w, max_w, 1)

    for i in range(6):
        for j in range(2):
            if Adj_N[i, j] == 1:
                Adj_N[i, j] = np.random.uniform(min_w, max_w, 1)
    # if Adj_M[0, 5] != 0:
    #     Adj_M[0, 5] = min_w

    return Adj_M, Adj_N, p_a, thy

def unified_model_corrected_confounders(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, 1, n_samples)
    u1 = np.random.normal(0, 1, n_samples)
    u2 = np.random.normal(0, 1, n_samples)
    u3 = np.random.normal(0, 1, n_samples)
    u4 = np.random.normal(0, 1, n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution
    Z2 = np.random.uniform(0, 2, n_samples)
    Z1 = np.random.gamma(2, 1, n_samples)

    # Identity matrix
    I_6 = np.eye(6)

    # Construct U and Z
    U = np.column_stack([uy, u1, u2, u3, u4,a])  # Shape: (n_samples, 7)
    Z = np.column_stack([Z1, Z2])  # Shape: (n_samples, 3)

    # Compute V
    I_minus_Adj_M = I_6 - Adj_M
    V = np.linalg.inv(I_minus_Adj_M) @ U.T + np.linalg.inv(I_minus_Adj_M) @ Adj_N @ Z.T  # Shape: (7, n_samples)

    # Extract variables
    Y, X1, X2, X3, X4, A = V
    Y = (Y-np.mean(Y))/np.std(Y)
    Y = np.where(Y > thy, 1, 0)
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1 :
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0]-min(Dkl[1:]))/Dkl[0]<0.2 or (Dkl_y[0]-min(Dkl_y[1:]))/Dkl_y[0]<0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins

def unified_model_corrected_confounders_same_con(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, var_y = 1):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, var_y, n_samples)
    u1 = np.random.normal(0, 1, n_samples)
    u2 = np.random.normal(0, 1, n_samples)
    u3 = np.random.normal(0, 1, n_samples)
    u4 = np.random.normal(0, 1, n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution
    Z2 = np.random.uniform(0, 2, n_samples)
    Z1 = np.random.uniform(0, 2, n_samples)

    # Identity matrix
    I_6 = np.eye(6)

    # Construct U and Z
    U = np.column_stack([uy, u1, u2, u3, u4,a])  # Shape: (n_samples, 7)
    Z = np.column_stack([Z1, Z2])  # Shape: (n_samples, 3)

    # Compute V
    I_minus_Adj_M = I_6 - Adj_M
    V = np.linalg.inv(I_minus_Adj_M) @ U.T + np.linalg.inv(I_minus_Adj_M) @ Adj_N @ Z.T  # Shape: (7, n_samples)

    # Extract variables
    Y, X1, X2, X3, X4, A = V

    # plot histogram for each variable
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # plt.show()
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # plt.show()
    # plt.hist(A, bins=100)
    # plt.title('A')
    # plt.show()


    Y = (Y-np.mean(Y))/np.std(Y)
    Y = np.where(Y > thy, 1, 0)
    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # for feature in features:
    #     df[feature],kk = pd.qcut(df[feature], q=6, labels=False, retbins=True, duplicates='drop')
        # print(df[feature].value_counts())
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1 :
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0]-min(Dkl[1:]))/Dkl[0]<0.2 or (Dkl_y[0]-min(Dkl_y[1:]))/Dkl_y[0]<0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins

def unified_model_corrected_standardized_SCM(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, var = 1):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, var, n_samples)
    u1 = np.random.normal(0, var, n_samples)
    u2 = np.random.normal(0, var, n_samples)
    u3 = np.random.normal(0, var, n_samples)
    u4 = np.random.normal(0, var, n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution
    Z2 = np.random.uniform(0, 2, n_samples)
    Z1 = np.random.uniform(0, 2, n_samples)
    #
    # Z1 = np.random.gamma(2, 1, n_samples) ###################
    # Z2 = np.random.gamma(2, 1, n_samples) ###################

    A = a
    # A_std = (A-np.mean(A))/np.std(A)
    A_std = A
    # Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    # Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    X4 = Adj_M[4, 5] * A_std + Adj_N[4, 0] * Z1 + Adj_N[4, 1] * Z2 + u4
    X4 = (X4-np.mean(X4))/np.std(X4)
    X3 = Adj_M[3, 4] * X4 + Adj_M[3, 5] * A_std + Adj_N[3, 0] * Z1 + Adj_N[3, 1] * Z2 + u3
    X3 = (X3-np.mean(X3))/np.std(X3)
    X2 = Adj_M[2, 3] * X3 + Adj_M[2, 4] * X4 + Adj_M[2, 5] * A_std + Adj_N[2, 0] * Z1 + Adj_N[2, 1] * Z2 + u2
    X2 = (X2-np.mean(X2))/np.std(X2)
    X1 = Adj_M[1, 2] * X2 + Adj_M[1, 3] * X3 + Adj_M[1, 4] * X4 + Adj_M[1, 5] * A_std + Adj_N[1, 0] * Z1 + Adj_N[1, 1] * Z2 + u1
    X1 = (X1-np.mean(X1))/np.std(X1)
    Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2 + uy
    Y = (Y-np.mean(Y))/np.std(Y)

    # # plot histogram of each variable
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # plt.show()
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # plt.show()



    Y = np.where(Y > thy, 1, 0)

    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # print(df.head(5))
    for feature in features:
        df[feature],kk = pd.qcut(df[feature], q=6, labels=False, retbins=True, duplicates='drop')
        # print(df[feature].value_counts())
    # for feature in features:
    #     df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1 :
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0]-min(Dkl[1:]))/Dkl[0]<0.2 or (Dkl_y[0]-min(Dkl_y[1:]))/Dkl_y[0]<0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins

def unified_model_corrected_standardized_SCM_N(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, var = 1):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, var, n_samples)
    u1 = np.random.normal(0, var, n_samples)
    u2 = np.random.normal(0, var, n_samples)
    u3 = np.random.normal(0, var, n_samples)
    u4 = np.random.normal(0, var, n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution
    Z2 = np.random.uniform(0, 2, n_samples)
    Z1 = np.random.uniform(0, 2, n_samples)
    #
    # Z1 = np.random.gamma(2, 1, n_samples) ###################
    # Z2 = np.random.gamma(2, 1, n_samples) ###################
    Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    A = a


    X4 = Adj_M[4, 5] * A + Adj_N[4, 0] * Z1 + Adj_N[4, 1] * Z2
    if Adj_N[4, 0] != 0 or Adj_N[4, 1] != 0:
        X4 = (X4-np.mean(X4))/np.std(X4)
    X4 = X4 + u4
    X3 = Adj_M[3, 4] * X4 + Adj_M[3, 5] * A + Adj_N[3, 0] * Z1 + Adj_N[3, 1] * Z2
    if Adj_M[3, 4] != 0 or Adj_N[3, 0] != 0 or Adj_N[3, 1] != 0:
        X3 = (X3-np.mean(X3))/np.std(X3)
    X3 = X3 + u3

    X2 = Adj_M[2, 3] * X3 + Adj_M[2, 4] * X4 + Adj_M[2, 5] * A + Adj_N[2, 0] * Z1 + Adj_N[2, 1] * Z2
    if Adj_M[2, 3] != 0 or Adj_M[2, 4] != 0 or Adj_N[2, 0] != 0 or Adj_N[2, 1] != 0:
        X2 = (X2-np.mean(X2))/np.std(X2)
    X2 = X2 + u2

    X1 = Adj_M[1, 2] * X2 + Adj_M[1, 3] * X3 + Adj_M[1, 4] * X4 + Adj_M[1, 5] * A + Adj_N[1, 0] * Z1 + Adj_N[1, 1] * Z2
    if Adj_M[1, 2] != 0 or Adj_M[1, 3] != 0 or Adj_M[1, 4] != 0 or Adj_N[1, 0] != 0 or Adj_N[1, 1] != 0:
        X1 = (X1-np.mean(X1))/np.std(X1)
    X1 = X1 + u1
    Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2
    if Adj_M[0, 1] != 0 or Adj_M[0, 2] != 0 or Adj_M[0, 3] != 0 or Adj_M[0, 4] != 0 or Adj_N[0, 0] != 0 or Adj_N[0, 1] != 0:
        Y = (Y-np.mean(Y))/np.std(Y)
    Y = Y + uy

    # # # plot histogram of each variable
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # plt.show()
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # plt.show()



    Y = np.where(Y > thy, 1, 0)

    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # print(df.head(5))
    for feature in features:
        df[feature],kk = pd.qcut(df[feature], q=6, labels=False, retbins=True, duplicates='drop')
        # print(df[feature].value_counts())
    # for feature in features:
    #     df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1 :
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0]-min(Dkl[1:]))/Dkl[0]<0.2 or (Dkl_y[0]-min(Dkl_y[1:]))/Dkl_y[0]<0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins

def unified_model_corrected_standardized_SCM_N_uniform(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, var_min = 0.1, var_max = 0.9):
    np.random.seed(time.time_ns() % (2 ** 32))
    var_v = np.random.uniform(var_min, var_max, 5)

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, var_v[0], n_samples)
    u1 = np.random.normal(0, var_v[1], n_samples)
    u2 = np.random.normal(0, var_v[2], n_samples)
    u3 = np.random.normal(0, var_v[3], n_samples)
    u4 = np.random.normal(0, var_v[4], n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution
    Z2 = np.random.uniform(0, 2, n_samples)
    Z1 = np.random.uniform(0, 2, n_samples)
    #
    # Z1 = np.random.gamma(2, 1, n_samples) ###################
    # Z2 = np.random.gamma(2, 1, n_samples) ###################
    Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    A = a


    X4 = Adj_M[4, 5] * A + Adj_N[4, 0] * Z1 + Adj_N[4, 1] * Z2
    if Adj_N[4, 0] != 0 or Adj_N[4, 1] != 0:
        X4 = (X4-np.mean(X4))/np.std(X4)
    X4 = X4 + u4
    X3 = Adj_M[3, 4] * X4 + Adj_M[3, 5] * A + Adj_N[3, 0] * Z1 + Adj_N[3, 1] * Z2
    if Adj_M[3, 4] != 0 or Adj_N[3, 0] != 0 or Adj_N[3, 1] != 0:
        X3 = (X3-np.mean(X3))/np.std(X3)
    X3 = X3 + u3

    X2 = Adj_M[2, 3] * X3 + Adj_M[2, 4] * X4 + Adj_M[2, 5] * A + Adj_N[2, 0] * Z1 + Adj_N[2, 1] * Z2
    if Adj_M[2, 3] != 0 or Adj_M[2, 4] != 0 or Adj_N[2, 0] != 0 or Adj_N[2, 1] != 0:
        X2 = (X2-np.mean(X2))/np.std(X2)
    X2 = X2 + u2

    X1 = Adj_M[1, 2] * X2 + Adj_M[1, 3] * X3 + Adj_M[1, 4] * X4 + Adj_M[1, 5] * A + Adj_N[1, 0] * Z1 + Adj_N[1, 1] * Z2
    if Adj_M[1, 2] != 0 or Adj_M[1, 3] != 0 or Adj_M[1, 4] != 0 or Adj_N[1, 0] != 0 or Adj_N[1, 1] != 0:
        X1 = (X1-np.mean(X1))/np.std(X1)
    X1 = X1 + u1
    Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2
    if Adj_M[0, 1] != 0 or Adj_M[0, 2] != 0 or Adj_M[0, 3] != 0 or Adj_M[0, 4] != 0 or Adj_N[0, 0] != 0 or Adj_N[0, 1] != 0:
        Y = (Y-np.mean(Y))/np.std(Y)
    Y = Y + uy

    # # # plot histogram of each variable
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # plt.show()
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # plt.show()



    Y = np.where(Y > thy, 1, 0)

    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # print(df.head(5))
    for feature in features:
        df[feature],kk = pd.qcut(df[feature], q=6, labels=False, retbins=True, duplicates='drop')
        # print(df[feature].value_counts())
    # for feature in features:
    #     df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1 :
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0]-min(Dkl[1:]))/Dkl[0]<0.2 or (Dkl_y[0]-min(Dkl_y[1:]))/Dkl_y[0]<0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins

def unified_model_corrected_standardized_SCM_uniform(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, var_min = 0.1, var_max = 0.9):
    np.random.seed(time.time_ns() % (2 ** 32))
    var_v = np.random.uniform(var_min, var_max, 5)
    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, var_v[0], n_samples)
    u1 = np.random.normal(0, var_v[1], n_samples)
    u2 = np.random.normal(0, var_v[2], n_samples)
    u3 = np.random.normal(0, var_v[3], n_samples)
    u4 = np.random.normal(0, var_v[4], n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution
    Z2 = np.random.uniform(0, 2, n_samples)
    Z1 = np.random.uniform(0, 2, n_samples)
    # #
    # Z1 = np.random.gamma(2, 1, n_samples) ###################
    # Z2 = np.random.gamma(2, 1, n_samples) ###################

    A = a
    # A_std = (A-np.mean(A))/np.std(A)
    A_std = A
    # Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    # Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    X4 = Adj_M[4, 5] * A_std + Adj_N[4, 0] * Z1 + Adj_N[4, 1] * Z2 + u4
    X4 = (X4-np.mean(X4))/np.std(X4)
    X3 = Adj_M[3, 4] * X4 + Adj_M[3, 5] * A_std + Adj_N[3, 0] * Z1 + Adj_N[3, 1] * Z2 + u3
    X3 = (X3-np.mean(X3))/np.std(X3)
    X2 = Adj_M[2, 3] * X3 + Adj_M[2, 4] * X4 + Adj_M[2, 5] * A_std + Adj_N[2, 0] * Z1 + Adj_N[2, 1] * Z2 + u2
    X2 = (X2-np.mean(X2))/np.std(X2)
    X1 = Adj_M[1, 2] * X2 + Adj_M[1, 3] * X3 + Adj_M[1, 4] * X4 + Adj_M[1, 5] * A_std + Adj_N[1, 0] * Z1 + Adj_N[1, 1] * Z2 + u1
    X1 = (X1-np.mean(X1))/np.std(X1)
    Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2 + uy
    Y = (Y-np.mean(Y))/np.std(Y)

    # # plot histogram of each variable
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # plt.show()
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # plt.show()



    Y = np.where(Y > thy, 1, 0)

    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # print(df.head(5))
    for feature in features:
        df[feature],kk = pd.qcut(df[feature], q=6, labels=False, retbins=True, duplicates='drop')
        # print(df[feature].value_counts())
    # for feature in features:
    #     df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1 :
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0]-min(Dkl[1:]))/Dkl[0]<0.2 or (Dkl_y[0]-min(Dkl_y[1:]))/Dkl_y[0]<0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins

def unified_model_corrected_standardized_SCM_N_uniform_var(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, var_v = []):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, var_v[0], n_samples)
    u1 = np.random.normal(0, var_v[1], n_samples)
    u2 = np.random.normal(0, var_v[2], n_samples)
    u3 = np.random.normal(0, var_v[3], n_samples)
    u4 = np.random.normal(0, var_v[4], n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution
    Z2 = np.random.uniform(0, 2, n_samples)
    Z1 = np.random.uniform(0, 2, n_samples)
    #
    # Z1 = np.random.gamma(2, 1, n_samples) ###################
    # Z2 = np.random.gamma(2, 1, n_samples) ###################
    Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    A = a


    X4 = Adj_M[4, 5] * A + Adj_N[4, 0] * Z1 + Adj_N[4, 1] * Z2
    if Adj_N[4, 0] != 0 or Adj_N[4, 1] != 0:
        X4 = (X4-np.mean(X4))/np.std(X4)
    X4 = X4 + u4
    X3 = Adj_M[3, 4] * X4 + Adj_M[3, 5] * A + Adj_N[3, 0] * Z1 + Adj_N[3, 1] * Z2
    if Adj_M[3, 4] != 0 or Adj_N[3, 0] != 0 or Adj_N[3, 1] != 0:
        X3 = (X3-np.mean(X3))/np.std(X3)
    X3 = X3 + u3

    X2 = Adj_M[2, 3] * X3 + Adj_M[2, 4] * X4 + Adj_M[2, 5] * A + Adj_N[2, 0] * Z1 + Adj_N[2, 1] * Z2
    if Adj_M[2, 3] != 0 or Adj_M[2, 4] != 0 or Adj_N[2, 0] != 0 or Adj_N[2, 1] != 0:
        X2 = (X2-np.mean(X2))/np.std(X2)
    X2 = X2 + u2

    X1 = Adj_M[1, 2] * X2 + Adj_M[1, 3] * X3 + Adj_M[1, 4] * X4 + Adj_M[1, 5] * A + Adj_N[1, 0] * Z1 + Adj_N[1, 1] * Z2
    if Adj_M[1, 2] != 0 or Adj_M[1, 3] != 0 or Adj_M[1, 4] != 0 or Adj_N[1, 0] != 0 or Adj_N[1, 1] != 0:
        X1 = (X1-np.mean(X1))/np.std(X1)
    X1 = X1 + u1
    Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2
    if Adj_M[0, 1] != 0 or Adj_M[0, 2] != 0 or Adj_M[0, 3] != 0 or Adj_M[0, 4] != 0 or Adj_N[0, 0] != 0 or Adj_N[0, 1] != 0:
        Y = (Y-np.mean(Y))/np.std(Y)
    Y = Y + uy

    # # # plot histogram of each variable
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # plt.show()
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # plt.show()



    Y = np.where(Y > thy, 1, 0)

    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # print(df.head(5))
    for feature in features:
        df[feature],kk = pd.qcut(df[feature], q=6, labels=False, retbins=True, duplicates='drop')
        # print(df[feature].value_counts())
    # for feature in features:
    #     df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1:
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0] - min(Dkl[1:])) / Dkl[0] < 0.2 or (Dkl_y[0] - min(Dkl_y[1:])) / Dkl_y[0] < 0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins


def unified_model_corrected_standardized_SCM_N_uniform_var_3con(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, var_v = []):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, var_v[0], n_samples)
    u1 = np.random.normal(0, var_v[1], n_samples)
    u2 = np.random.normal(0, var_v[2], n_samples)
    u3 = np.random.normal(0, var_v[3], n_samples)
    u4 = np.random.normal(0, var_v[4], n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution
    Z3 = np.random.uniform(0, 2, n_samples)
    Z2 = np.random.uniform(0, 2, n_samples)
    Z1 = np.random.uniform(0, 2, n_samples)
    #
    # Z1 = np.random.gamma(2, 1, n_samples) ###################
    # Z2 = np.random.gamma(2, 1, n_samples) ###################
    Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    Z3 = (Z3-np.mean(Z3))/np.std(Z3)
    A = a


    X4 = Adj_M[4, 5] * A + Adj_N[4, 0] * Z1 + Adj_N[4, 1] * Z2 + Adj_N[4, 2] * Z3
    if Adj_N[4, 0] != 0 or Adj_N[4, 1] != 0:
        X4 = (X4-np.mean(X4))/np.std(X4)
    X4 = X4 + u4
    X3 = Adj_M[3, 4] * X4 + Adj_M[3, 5] * A + Adj_N[3, 0] * Z1 + Adj_N[3, 1] * Z2 + Adj_N[3, 2] * Z3
    if Adj_M[3, 4] != 0 or Adj_N[3, 0] != 0 or Adj_N[3, 1] != 0:
        X3 = (X3-np.mean(X3))/np.std(X3)
    X3 = X3 + u3

    X2 = Adj_M[2, 3] * X3 + Adj_M[2, 4] * X4 + Adj_M[2, 5] * A + Adj_N[2, 0] * Z1 + Adj_N[2, 1] * Z2 + Adj_N[2, 2] * Z3
    if Adj_M[2, 3] != 0 or Adj_M[2, 4] != 0 or Adj_N[2, 0] != 0 or Adj_N[2, 1] != 0:
        X2 = (X2-np.mean(X2))/np.std(X2)
    X2 = X2 + u2

    X1 = Adj_M[1, 2] * X2 + Adj_M[1, 3] * X3 + Adj_M[1, 4] * X4 + Adj_M[1, 5] * A + Adj_N[1, 0] * Z1 + Adj_N[1, 1] * Z2 + Adj_N[1, 2] * Z3
    if Adj_M[1, 2] != 0 or Adj_M[1, 3] != 0 or Adj_M[1, 4] != 0 or Adj_N[1, 0] != 0 or Adj_N[1, 1] != 0:
        X1 = (X1-np.mean(X1))/np.std(X1)
    X1 = X1 + u1
    Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2 + Adj_N[0, 2] * Z3
    if Adj_M[0, 1] != 0 or Adj_M[0, 2] != 0 or Adj_M[0, 3] != 0 or Adj_M[0, 4] != 0 or Adj_N[0, 0] != 0 or Adj_N[0, 1] != 0:
        Y = (Y-np.mean(Y))/np.std(Y)
    Y = Y + uy

    # # # plot histogram of each variable
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # plt.show()
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # plt.show()



    Y = np.where(Y > thy, 1, 0)

    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # print(df.head(5))
    for feature in features:
        df[feature],kk = pd.qcut(df[feature], q=6, labels=False, retbins=True, duplicates='drop')
        # print(df[feature].value_counts())
    # for feature in features:
    #     df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1:
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0] - min(Dkl[1:])) / Dkl[0] < 0.2 or (Dkl_y[0] - min(Dkl_y[1:])) / Dkl_y[0] < 0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins


def unified_model_corrected_standardized_SCM_N_uniform_var_3con_updated(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, var_v = []):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, 1, n_samples)
    u1 = np.random.normal(0, 1, n_samples)
    u2 = np.random.normal(0, 1, n_samples)
    u3 = np.random.normal(0, 1, n_samples)
    u4 = np.random.normal(0, 1, n_samples)
    a = np.random.binomial(1, p_a, n_samples)
    a_binary = a
    a = a + np.random.normal(0, 0.2, n_samples)


    # uniform distribution, Gaussian distribution, Gamma distribution
    Z3 = np.random.uniform(0, 2, n_samples)
    Z2 = np.random.uniform(0, 2, n_samples)
    Z1 = np.random.uniform(0, 2, n_samples)

    # standarize a and Z1, Z2, Z3
    a = (a-np.mean(a))/np.std(a)
    Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    Z3 = (Z3-np.mean(Z3))/np.std(Z3)

    #
    # Z1 = np.random.gamma(2, 1, n_samples) ###################
    # Z2 = np.random.gamma(2, 1, n_samples) ###################
    Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    Z3 = (Z3-np.mean(Z3))/np.std(Z3)
    A = a


    X4 = Adj_M[4, 5] * A + Adj_N[4, 0] * Z1 + Adj_N[4, 1] * Z2 + Adj_N[4, 2] * Z3
    if np.std(X4) != 0:
        X4 = (X4-np.mean(X4))/np.std(X4)
        X4 = math.sqrt(var_v[4])* X4 + math.sqrt(1-var_v[4])*u4
    else:
        X4 = u4
    X3 = Adj_M[3, 4] * X4 + Adj_M[3, 5] * A + Adj_N[3, 0] * Z1 + Adj_N[3, 1] * Z2 + Adj_N[3, 2] * Z3
    if np.std(X3) != 0:
        X3 = (X3-np.mean(X3))/np.std(X3)
        X3 = math.sqrt(var_v[3])* X3 + math.sqrt(1-var_v[3])*u3
    else:
        X3 = u3


    X2 = Adj_M[2, 3] * X3 + Adj_M[2, 4] * X4 + Adj_M[2, 5] * A + Adj_N[2, 0] * Z1 + Adj_N[2, 1] * Z2 + Adj_N[2, 2] * Z3
    if np.std(X2) != 0:
        X2 = (X2-np.mean(X2))/np.std(X2)
        X2 = math.sqrt(var_v[2])* X2 + math.sqrt(1-var_v[2])*u2
    else:
        X2 = u2

    X1 = Adj_M[1, 2] * X2 + Adj_M[1, 3] * X3 + Adj_M[1, 4] * X4 + Adj_M[1, 5] * A + Adj_N[1, 0] * Z1 + Adj_N[1, 1] * Z2 + Adj_N[1, 2] * Z3
    if np.std(X1) != 0:
        X1 = (X1-np.mean(X1))/np.std(X1)
        X1 = math.sqrt(var_v[1])* X1 + math.sqrt(1-var_v[1])*u1
    else:
        X1 = u1

    Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2 + Adj_N[0, 2] * Z3
    if np.std(Y) != 0:
        Y = (Y-np.mean(Y))/np.std(Y)
        Y = math.sqrt(var_v[0])* Y + math.sqrt(1-var_v[0])*uy
    else:
        Y = uy

    # # # plot histogram of each variable
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # print("X1: ", np.mean(X1), np.std(X1))
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # print("X2: ", np.mean(X2), np.std(X2))
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # print("X3: ", np.mean(X3), np.std(X3))
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # print("X4: ", np.mean(X4), np.std(X4))
    # plt.show()
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # print("Y: ", np.mean(Y), np.std(Y))
    # plt.show()
    #
    # exit()

    Y = np.where(Y > thy, 1, 0)

    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': a_binary,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # print(df.head(5))
    for feature in features:
        df[feature],kk = pd.qcut(df[feature], q=6, labels=False, retbins=True, duplicates='drop')
        # print(df[feature].value_counts())
    # for feature in features:
    #     df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1:
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0] - min(Dkl[1:])) / Dkl[0] < 0.2 or (Dkl_y[0] - min(Dkl_y[1:])) / Dkl_y[0] < 0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins




def unified_model_corrected_standardized_SCM_uniform_var(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, var_v = []):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, var_v[0], n_samples)
    u1 = np.random.normal(0, var_v[1], n_samples)
    u2 = np.random.normal(0, var_v[2], n_samples)
    u3 = np.random.normal(0, var_v[3], n_samples)
    u4 = np.random.normal(0, var_v[4], n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution
    Z2 = np.random.uniform(0, 2, n_samples)
    Z1 = np.random.uniform(0, 2, n_samples)
    # #
    # Z1 = np.random.gamma(2, 1, n_samples) ###################
    # Z2 = np.random.gamma(2, 1, n_samples) ###################

    A = a
    # A_std = (A-np.mean(A))/np.std(A)
    A_std = A
    # Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    # Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    X4 = Adj_M[4, 5] * A_std + Adj_N[4, 0] * Z1 + Adj_N[4, 1] * Z2 + u4
    X4 = (X4-np.mean(X4))/np.std(X4)
    X3 = Adj_M[3, 4] * X4 + Adj_M[3, 5] * A_std + Adj_N[3, 0] * Z1 + Adj_N[3, 1] * Z2 + u3
    X3 = (X3-np.mean(X3))/np.std(X3)
    X2 = Adj_M[2, 3] * X3 + Adj_M[2, 4] * X4 + Adj_M[2, 5] * A_std + Adj_N[2, 0] * Z1 + Adj_N[2, 1] * Z2 + u2
    X2 = (X2-np.mean(X2))/np.std(X2)
    X1 = Adj_M[1, 2] * X2 + Adj_M[1, 3] * X3 + Adj_M[1, 4] * X4 + Adj_M[1, 5] * A_std + Adj_N[1, 0] * Z1 + Adj_N[1, 1] * Z2 + u1
    X1 = (X1-np.mean(X1))/np.std(X1)
    Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2 + uy
    Y = (Y-np.mean(Y))/np.std(Y)

    # plot histogram of each variable
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # plt.show()
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # plt.show()



    Y = np.where(Y > thy, 1, 0)

    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # print(df.head(5))
    for feature in features:
        df[feature],kk = pd.qcut(df[feature], q=6, labels=False, retbins=True, duplicates='drop')
        # print(df[feature].value_counts())
    # for feature in features:
    #     df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1 :
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0]-min(Dkl[1:]))/Dkl[0]<0.2 or (Dkl_y[0]-min(Dkl_y[1:]))/Dkl_y[0]<0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins


def unified_model_corrected_standardized_SCM_uniform_var_3con_EWQ(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, var_v = []):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, var_v[0], n_samples)
    u1 = np.random.normal(0, var_v[1], n_samples)
    u2 = np.random.normal(0, var_v[2], n_samples)
    u3 = np.random.normal(0, var_v[3], n_samples)
    u4 = np.random.normal(0, var_v[4], n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution

    Z1 = np.random.uniform(0, 2, n_samples)
    Z2 = np.random.uniform(0, 2, n_samples)
    Z3 = np.random.uniform(0, 2, n_samples)
    # #
    # Z1 = np.random.gamma(2, 1, n_samples) ###################
    # Z2 = np.random.gamma(2, 1, n_samples) ###################

    A = a
    # A_std = (A-np.mean(A))/np.std(A)
    A_std = A
    # Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    # Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    X4 = Adj_M[4, 5] * A_std + Adj_N[4, 0] * Z1 + Adj_N[4, 1] * Z2 + + Adj_N[4, 2] * Z3 + u4
    X4 = (X4-np.mean(X4))/np.std(X4)
    X3 = Adj_M[3, 4] * X4 + Adj_M[3, 5] * A_std + Adj_N[3, 0] * Z1 + Adj_N[3, 1] * Z2 + Adj_N[3, 2] * Z3 + u3
    X3 = (X3-np.mean(X3))/np.std(X3)
    X2 = Adj_M[2, 3] * X3 + Adj_M[2, 4] * X4 + Adj_M[2, 5] * A_std + Adj_N[2, 0] * Z1 + Adj_N[2, 1] * Z2 + Adj_N[2, 2] * Z3 + u2
    X2 = (X2-np.mean(X2))/np.std(X2)
    X1 = Adj_M[1, 2] * X2 + Adj_M[1, 3] * X3 + Adj_M[1, 4] * X4 + Adj_M[1, 5] * A_std + Adj_N[1, 0] * Z1 + Adj_N[1, 1] * Z2 + Adj_N[1, 2] * Z3 + u1
    X1 = (X1-np.mean(X1))/np.std(X1)
    Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2 + Adj_N[0, 2] * Z3 + uy
    Y = (Y-np.mean(Y))/np.std(Y)

    # plot histogram of each variable
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # plt.show()
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # plt.show()



    Y = np.where(Y > thy, 1, 0)

    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # print(df.head(5))
    for feature in features:
        df[feature] = pd.cut(df[feature], bins=6, labels=False)

        # print(df[feature].value_counts())
    # for feature in features:
    #     df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1 :
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0]-min(Dkl[1:]))/Dkl[0]<0.2 or (Dkl_y[0]-min(Dkl_y[1:]))/Dkl_y[0]<0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins



def unified_model_corrected_standardized_SCM_uniform_var_3con(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, var_v = []):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, var_v[0], n_samples)
    u1 = np.random.normal(0, var_v[1], n_samples)
    u2 = np.random.normal(0, var_v[2], n_samples)
    u3 = np.random.normal(0, var_v[3], n_samples)
    u4 = np.random.normal(0, var_v[4], n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution

    Z1 = np.random.uniform(0, 2, n_samples)
    Z2 = np.random.uniform(0, 2, n_samples)
    Z3 = np.random.uniform(0, 2, n_samples)
    # #
    # Z1 = np.random.gamma(2, 1, n_samples) ###################
    # Z2 = np.random.gamma(2, 1, n_samples) ###################

    A = a
    # A_std = (A-np.mean(A))/np.std(A)
    A_std = A
    # Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    # Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    X4 = Adj_M[4, 5] * A_std + Adj_N[4, 0] * Z1 + Adj_N[4, 1] * Z2 + + Adj_N[4, 2] * Z3 + u4
    X4 = (X4-np.mean(X4))/np.std(X4)
    X3 = Adj_M[3, 4] * X4 + Adj_M[3, 5] * A_std + Adj_N[3, 0] * Z1 + Adj_N[3, 1] * Z2 + Adj_N[3, 2] * Z3 + u3
    X3 = (X3-np.mean(X3))/np.std(X3)
    X2 = Adj_M[2, 3] * X3 + Adj_M[2, 4] * X4 + Adj_M[2, 5] * A_std + Adj_N[2, 0] * Z1 + Adj_N[2, 1] * Z2 + Adj_N[2, 2] * Z3 + u2
    X2 = (X2-np.mean(X2))/np.std(X2)
    X1 = Adj_M[1, 2] * X2 + Adj_M[1, 3] * X3 + Adj_M[1, 4] * X4 + Adj_M[1, 5] * A_std + Adj_N[1, 0] * Z1 + Adj_N[1, 1] * Z2 + Adj_N[1, 2] * Z3 + u1
    X1 = (X1-np.mean(X1))/np.std(X1)
    Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2 + Adj_N[0, 2] * Z3 + uy
    Y = (Y-np.mean(Y))/np.std(Y)

    # plot histogram of each variable
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # plt.show()
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # plt.show()



    Y = np.where(Y > thy, 1, 0)

    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # print(df.head(5))
    for feature in features:
        df[feature],kk = pd.qcut(df[feature], q=6, labels=False, retbins=True, duplicates='drop')
        # print(df[feature].value_counts())
    # for feature in features:
    #     df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1 :
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0]-min(Dkl[1:]))/Dkl[0]<0.2 or (Dkl_y[0]-min(Dkl_y[1:]))/Dkl_y[0]<0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins



def unified_model_corrected_SCM_uniform_var_3con_EFQ(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, var_v = []):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, 1, n_samples)
    u1 = np.random.normal(0, 1, n_samples)
    u2 = np.random.normal(0, 1, n_samples)
    u3 = np.random.normal(0, 1, n_samples)
    u4 = np.random.normal(0, 1, n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution

    Z1 = np.random.uniform(0, 2, n_samples)
    Z2 = np.random.uniform(0, 2, n_samples)
    Z3 = np.random.uniform(0, 2, n_samples)
    # #
    # Z1 = np.random.gamma(2, 1, n_samples) ###################
    # Z2 = np.random.gamma(2, 1, n_samples) ###################

    A = a
    # A_std = (A-np.mean(A))/np.std(A)
    A_std = A
    # Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    # Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    X4 = Adj_M[4, 5] * A_std + Adj_N[4, 0] * Z1 + Adj_N[4, 1] * Z2 + + Adj_N[4, 2] * Z3 + u4
    # X4 = (X4-np.mean(X4))/np.std(X4)
    X3 = Adj_M[3, 4] * X4 + Adj_M[3, 5] * A_std + Adj_N[3, 0] * Z1 + Adj_N[3, 1] * Z2 + Adj_N[3, 2] * Z3 + u3
    # X3 = (X3-np.mean(X3))/np.std(X3)
    X2 = Adj_M[2, 3] * X3 + Adj_M[2, 4] * X4 + Adj_M[2, 5] * A_std + Adj_N[2, 0] * Z1 + Adj_N[2, 1] * Z2 + Adj_N[2, 2] * Z3 + u2
    # X2 = (X2-np.mean(X2))/np.std(X2)
    X1 = Adj_M[1, 2] * X2 + Adj_M[1, 3] * X3 + Adj_M[1, 4] * X4 + Adj_M[1, 5] * A_std + Adj_N[1, 0] * Z1 + Adj_N[1, 1] * Z2 + Adj_N[1, 2] * Z3 + u1
    # X1 = (X1-np.mean(X1))/np.std(X1)
    Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2 + Adj_N[0, 2] * Z3 + uy
    Y = (Y-np.mean(Y))/np.std(Y)

    # plot histogram of each variable
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # plt.show()
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # plt.show()



    Y = np.where(Y > thy, 1, 0)

    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # print(df.head(5))
    for feature in features:
        df[feature],kk = pd.qcut(df[feature], q=6, labels=False, retbins=True, duplicates='drop')
        # print(df[feature].value_counts())
    # for feature in features:
    #     df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1 :
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0]-min(Dkl[1:]))/Dkl[0]<0.2 or (Dkl_y[0]-min(Dkl_y[1:]))/Dkl_y[0]<0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins


def unified_model_corrected_SCM_uniform_var_3con_EFQ_with_acc(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, lam = 0):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, 1, n_samples)
    u1 = np.random.normal(0, 1, n_samples)
    u2 = np.random.normal(0, 1, n_samples)
    u3 = np.random.normal(0, 1, n_samples)
    u4 = np.random.normal(0, 1, n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution

    Z1 = np.random.uniform(0, 2, n_samples)
    Z2 = np.random.uniform(0, 2, n_samples)
    Z3 = np.random.uniform(0, 2, n_samples)
    # #
    # Z1 = np.random.gamma(2, 1, n_samples) ###################
    # Z2 = np.random.gamma(2, 1, n_samples) ###################

    A = a
    # A_std = (A-np.mean(A))/np.std(A)
    A_std = A
    # Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    # Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    X4 = Adj_M[4, 5] * A_std + Adj_N[4, 0] * Z1 + Adj_N[4, 1] * Z2 + + Adj_N[4, 2] * Z3 + u4
    # X4 = (X4-np.mean(X4))/np.std(X4)
    X3 = Adj_M[3, 4] * X4 + Adj_M[3, 5] * A_std + Adj_N[3, 0] * Z1 + Adj_N[3, 1] * Z2 + Adj_N[3, 2] * Z3 + u3
    # X3 = (X3-np.mean(X3))/np.std(X3)
    X2 = Adj_M[2, 3] * X3 + Adj_M[2, 4] * X4 + Adj_M[2, 5] * A_std + Adj_N[2, 0] * Z1 + Adj_N[2, 1] * Z2 + Adj_N[2, 2] * Z3 + u2
    # X2 = (X2-np.mean(X2))/np.std(X2)
    X1 = Adj_M[1, 2] * X2 + Adj_M[1, 3] * X3 + Adj_M[1, 4] * X4 + Adj_M[1, 5] * A_std + Adj_N[1, 0] * Z1 + Adj_N[1, 1] * Z2 + Adj_N[1, 2] * Z3 + u1
    # X1 = (X1-np.mean(X1))/np.std(X1)
    Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2 + Adj_N[0, 2] * Z3
    Y = math.sqrt(lam)*(Y-np.mean(Y))/np.std(Y) + (1-math.sqrt(lam))*uy

    # plot histogram of each variable
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # plt.show()
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # plt.show()



    Y = np.where(Y > thy, 1, 0)

    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # print(df.head(5))
    for feature in features:
        df[feature],kk = pd.qcut(df[feature], q=6, labels=False, retbins=True, duplicates='drop')
        # print(df[feature].value_counts())
    # for feature in features:
    #     df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1 :
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0]-min(Dkl[1:]))/Dkl[0]<0.2 or (Dkl_y[0]-min(Dkl_y[1:]))/Dkl_y[0]<0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins

def unified_model_corrected_SCM_uniform_var_3con_EFQ_with_acc_con_type(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, lam = 0, con_type = 0):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, 1, n_samples)
    u1 = np.random.normal(0, 1, n_samples)
    u2 = np.random.normal(0, 1, n_samples)
    u3 = np.random.normal(0, 1, n_samples)
    u4 = np.random.normal(0, 1, n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution

    Z1 = np.random.uniform(0, 2, n_samples)
    # Z2 = np.random.uniform(0, 2, n_samples)
    Z3 = np.random.uniform(0, 2, n_samples)
    # #
    # Z1 = np.random.gamma(2, 1, n_samples) ###################
    Z2 = np.random.gamma(2, 1, n_samples) ###################

    if con_type == 1:
        zz1 = Z2
        Z2 = Z1
        Z1 = zz1

    A = a
    # A_std = (A-np.mean(A))/np.std(A)
    A_std = A
    # Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    # Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    X4 = Adj_M[4, 5] * A_std + Adj_N[4, 0] * Z1 + Adj_N[4, 1] * Z2 + + Adj_N[4, 2] * Z3 + u4
    # X4 = (X4-np.mean(X4))/np.std(X4)
    X3 = Adj_M[3, 4] * X4 + Adj_M[3, 5] * A_std + Adj_N[3, 0] * Z1 + Adj_N[3, 1] * Z2 + Adj_N[3, 2] * Z3 + u3
    # X3 = (X3-np.mean(X3))/np.std(X3)
    X2 = Adj_M[2, 3] * X3 + Adj_M[2, 4] * X4 + Adj_M[2, 5] * A_std + Adj_N[2, 0] * Z1 + Adj_N[2, 1] * Z2 + Adj_N[2, 2] * Z3 + u2
    # X2 = (X2-np.mean(X2))/np.std(X2)
    X1 = Adj_M[1, 2] * X2 + Adj_M[1, 3] * X3 + Adj_M[1, 4] * X4 + Adj_M[1, 5] * A_std + Adj_N[1, 0] * Z1 + Adj_N[1, 1] * Z2 + Adj_N[1, 2] * Z3 + u1
    # X1 = (X1-np.mean(X1))/np.std(X1)
    Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2 + Adj_N[0, 2] * Z3
    Y = math.sqrt(lam)*(Y-np.mean(Y))/np.std(Y) + (1-math.sqrt(lam))*uy

    # plot histogram of each variable
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # plt.show()
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # plt.show()



    Y = np.where(Y > thy, 1, 0)

    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # print(df.head(5))
    for feature in features:
        df[feature],kk = pd.qcut(df[feature], q=6, labels=False, retbins=True, duplicates='drop')
        # print(df[feature].value_counts())
    # for feature in features:
    #     df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1 :
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0]-min(Dkl[1:]))/Dkl[0]<0.2 or (Dkl_y[0]-min(Dkl_y[1:]))/Dkl_y[0]<0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins

def unified_model_corrected_SCM_uniform_var_3con_EWQ(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, var_v = []):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, 1, n_samples)
    u1 = np.random.normal(0, 1, n_samples)
    u2 = np.random.normal(0, 1, n_samples)
    u3 = np.random.normal(0, 1, n_samples)
    u4 = np.random.normal(0, 1, n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution

    Z1 = np.random.uniform(0, 2, n_samples)
    Z2 = np.random.uniform(0, 2, n_samples)
    Z3 = np.random.uniform(0, 2, n_samples)
    # #
    # Z1 = np.random.gamma(2, 1, n_samples) ###################
    # Z2 = np.random.gamma(2, 1, n_samples) ###################

    A = a
    # A_std = (A-np.mean(A))/np.std(A)
    A_std = A
    # Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    # Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    X4 = Adj_M[4, 5] * A_std + Adj_N[4, 0] * Z1 + Adj_N[4, 1] * Z2 + + Adj_N[4, 2] * Z3 + u4
    # X4 = (X4-np.mean(X4))/np.std(X4)
    X3 = Adj_M[3, 4] * X4 + Adj_M[3, 5] * A_std + Adj_N[3, 0] * Z1 + Adj_N[3, 1] * Z2 + Adj_N[3, 2] * Z3 + u3
    # X3 = (X3-np.mean(X3))/np.std(X3)
    X2 = Adj_M[2, 3] * X3 + Adj_M[2, 4] * X4 + Adj_M[2, 5] * A_std + Adj_N[2, 0] * Z1 + Adj_N[2, 1] * Z2 + Adj_N[2, 2] * Z3 + u2
    # X2 = (X2-np.mean(X2))/np.std(X2)
    X1 = Adj_M[1, 2] * X2 + Adj_M[1, 3] * X3 + Adj_M[1, 4] * X4 + Adj_M[1, 5] * A_std + Adj_N[1, 0] * Z1 + Adj_N[1, 1] * Z2 + Adj_N[1, 2] * Z3 + u1
    # X1 = (X1-np.mean(X1))/np.std(X1)
    Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2 + Adj_N[0, 2] * Z3 + uy
    Y = (Y-np.mean(Y))/np.std(Y)

    # plot histogram of each variable
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # plt.show()
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # plt.show()



    Y = np.where(Y > thy, 1, 0)

    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # print(df.head(5))
    for feature in features:
        df[feature], kk = pd.cut(df[feature], bins=6, labels=False, retbins=True, duplicates='drop')

        # print(df[feature].value_counts())
    # for feature in features:
    #     df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1 :
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0]-min(Dkl[1:]))/Dkl[0]<0.2 or (Dkl_y[0]-min(Dkl_y[1:]))/Dkl_y[0]<0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins


def unified_model_corrected_standardized_SCM_uniform_var_sp(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, var_v = []):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, var_v[0], n_samples)
    u1 = np.random.normal(0, 1, n_samples)
    u2 = np.random.normal(0, 1, n_samples)
    u3 = np.random.normal(0, 1, n_samples)
    u4 = np.random.normal(0, 1, n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution
    Z2 = np.random.uniform(0, 2, n_samples)
    Z1 = np.random.uniform(0, 2, n_samples)
    # #
    # Z1 = np.random.gamma(2, 1, n_samples) ###################
    # Z2 = np.random.gamma(2, 1, n_samples) ###################

    A = a
    # A_std = (A-np.mean(A))/np.std(A)
    A_std = A
    # Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    # Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    X4 = Adj_M[4, 5] * A_std + Adj_N[4, 0] * Z1 + Adj_N[4, 1] * Z2 + u4
    X4 = (X4-np.mean(X4))/np.std(X4)
    X3 = Adj_M[3, 4] * X4 + Adj_M[3, 5] * A_std + Adj_N[3, 0] * Z1 + Adj_N[3, 1] * Z2 + u3
    X3 = (X3-np.mean(X3))/np.std(X3)
    X2 = Adj_M[2, 3] * X3 + Adj_M[2, 4] * X4 + Adj_M[2, 5] * A_std + Adj_N[2, 0] * Z1 + Adj_N[2, 1] * Z2 + u2
    X2 = (X2-np.mean(X2))/np.std(X2)
    X1 = Adj_M[1, 2] * X2 + Adj_M[1, 3] * X3 + Adj_M[1, 4] * X4 + Adj_M[1, 5] * A_std + Adj_N[1, 0] * Z1 + Adj_N[1, 1] * Z2 + u1
    X1 = (X1-np.mean(X1))/np.std(X1)

    Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2
    if Adj_M[0, 1] != 0 or Adj_M[0, 2] != 0 or Adj_M[0, 3] != 0 or Adj_M[0, 4] != 0 or Adj_N[0, 0] != 0 or Adj_N[0, 1] != 0:
        Y = (Y - np.mean(Y)) / np.std(Y)
    Y = Y + uy

    # Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2 + uy
    Y = (Y-np.mean(Y))/np.std(Y)

    # plot histogram of each variable
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # plt.show()
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # plt.show()



    Y = np.where(Y > thy, 1, 0)

    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # print(df.head(5))
    for feature in features:
        df[feature],kk = pd.qcut(df[feature], q=6, labels=False, retbins=True, duplicates='drop')
        # print(df[feature].value_counts())
    # for feature in features:
    #     df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1 :
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0]-min(Dkl[1:]))/Dkl[0]<0.2 or (Dkl_y[0]-min(Dkl_y[1:]))/Dkl_y[0]<0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins

def unified_model_corrected_standardized_SCM_uniform_var_sp_2(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5, var_v = [],v_f = 0.75):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, var_v[0], n_samples)
    u1 = np.random.normal(0, v_f, n_samples)
    u2 = np.random.normal(0, v_f, n_samples)
    u3 = np.random.normal(0, v_f, n_samples)
    u4 = np.random.normal(0, v_f, n_samples)
    a = np.random.binomial(1, p_a, n_samples)

    # uniform distribution, Gaussian distribution, Gamma distribution
    Z2 = np.random.uniform(0, 2, n_samples)
    Z1 = np.random.uniform(0, 2, n_samples)
    # #
    # Z1 = np.random.gamma(2, 1, n_samples) ###################
    # Z2 = np.random.gamma(2, 1, n_samples) ###################

    A = a
    # A_std = (A-np.mean(A))/np.std(A)
    A_std = A
    # Z1 = (Z1-np.mean(Z1))/np.std(Z1)
    # Z2 = (Z2-np.mean(Z2))/np.std(Z2)
    X4 = Adj_M[4, 5] * A_std + Adj_N[4, 0] * Z1 + Adj_N[4, 1] * Z2 + u4
    X4 = (X4-np.mean(X4))/np.std(X4)
    X3 = Adj_M[3, 4] * X4 + Adj_M[3, 5] * A_std + Adj_N[3, 0] * Z1 + Adj_N[3, 1] * Z2 + u3
    X3 = (X3-np.mean(X3))/np.std(X3)
    X2 = Adj_M[2, 3] * X3 + Adj_M[2, 4] * X4 + Adj_M[2, 5] * A_std + Adj_N[2, 0] * Z1 + Adj_N[2, 1] * Z2 + u2
    X2 = (X2-np.mean(X2))/np.std(X2)
    X1 = Adj_M[1, 2] * X2 + Adj_M[1, 3] * X3 + Adj_M[1, 4] * X4 + Adj_M[1, 5] * A_std + Adj_N[1, 0] * Z1 + Adj_N[1, 1] * Z2 + u1
    X1 = (X1-np.mean(X1))/np.std(X1)

    Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2
    if Adj_M[0, 1] != 0 or Adj_M[0, 2] != 0 or Adj_M[0, 3] != 0 or Adj_M[0, 4] != 0 or Adj_N[0, 0] != 0 or Adj_N[0, 1] != 0:
        Y = (Y - np.mean(Y)) / np.std(Y)
    Y = Y + uy

    # Y = Adj_M[0, 1] * X1 + Adj_M[0, 2] * X2 + Adj_M[0, 3] * X3 + Adj_M[0, 4] * X4 + Adj_M[0, 5] * A + Adj_N[0, 0] * Z1 + Adj_N[0, 1] * Z2 + uy
    Y = (Y-np.mean(Y))/np.std(Y)

    # plot histogram of each variable
    # plt.hist(X1, bins=100)
    # plt.title('X1')
    # plt.show()
    # plt.hist(X2, bins=100)
    # plt.title('X2')
    # plt.show()
    # plt.hist(X3, bins=100)
    # plt.title('X3')
    # plt.show()
    # plt.hist(X4, bins=100)
    # plt.title('X4')
    # plt.show()
    # plt.hist(Y, bins=100)
    # plt.title('Y')
    # plt.show()



    Y = np.where(Y > thy, 1, 0)

    df = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    # print(df.head(5))
    for feature in features:
        df[feature],kk = pd.qcut(df[feature], q=6, labels=False, retbins=True, duplicates='drop')
        # print(df[feature].value_counts())
    # for feature in features:
    #     df[feature] = pd.cut(df[feature], bins=6, labels=False)
    bins = (6, 6, 6, 6, 2, 2)

    if parameters[0] == 1:
        # check if DP and EO are larger than 0.1
        accuracy, cross_entropy_loss, DP, DP_prob, \
        I, I_prob, NMI, NMI_prob, Dkl, Dkl_prob, TV, TV_prob, \
        I_y, I_prob_y, NMI_y, NMI_prob_y, Dkl_y, Dkl_prob_y, TV_y, TV_prob_y, \
        I_y1, I_prob_y1, NMI_y1, NMI_prob_y1, Dkl_y1, Dkl_prob_y1, TV_y1, TV_prob_y1, \
        I_y0, I_prob_y0, NMI_y0, NMI_prob_y0, Dkl_y0, Dkl_prob_y0, TV_y0, TV_prob_y0, \
        Original_DP, PR = classify_all_and_drop_sequentially_replace_with_mean_new(
            df, features, sensitive_attribute, output, bins=2, seed=seed, standardize=True, use_A=True, N=number_of_neurons, LR=LR_classifier, regularizations=10000)

        # print("PR: ", PR)
        if Dkl[0] < 0.1 or Dkl_y[0] < 0.1 :
            # print("Model is almost fair.")
            features = [1]
            return df, features, sensitive_attribute, output, bins
        if (Dkl[0]-min(Dkl[1:]))/Dkl[0]<0.2 or (Dkl_y[0]-min(Dkl_y[1:]))/Dkl_y[0]<0.2:
            features = [1]
            return df, features, sensitive_attribute, output, bins

    return df, features, sensitive_attribute, output, bins


def plot_causal_graph(Adj_M, Adj_N, feature_names, confounder_names, edge_labels=True, layout='circular',path="",name=""):
    """
    Plot the causal graph defined by Adj_M and Adj_N.

    Parameters:
    -----------
    Adj_M : np.ndarray
        Adjacency matrix defining edges between features (nodes).
    Adj_N : np.ndarray
        Adjacency matrix defining edges from confounders to features.
    feature_names : list of str
        Names of the features (nodes corresponding to Adj_M rows/columns).
    confounder_names : list of str
        Names of the confounders (nodes corresponding to Adj_N rows).
    edge_labels : bool, optional
        Whether to display edge weights as labels. Default is True.
    layout : str, optional
        Graph layout for visualization. Options are 'spring', 'circular', or 'kamada_kawai'. Default is 'spring'.

    Returns:
    --------
    None
    """
    # Validate dimensions
    if Adj_M.shape[0] != Adj_M.shape[1] or Adj_M.shape[0] != len(feature_names):
        raise ValueError("Adj_M dimensions must match the number of feature names.")
    if Adj_N.shape[1] != len(confounder_names):
        raise ValueError("Number of columns in Adj_N must match the number of confounder names.")

    G = nx.DiGraph()

    # Add feature nodes
    G.add_nodes_from(feature_names)

    # Add edges from Adj_M (feature-to-feature connections)
    num_features = Adj_M.shape[0]
    for i in range(num_features):
        for j in range(num_features):
            if Adj_M[i, j] != 0:
                G.add_edge(feature_names[j], feature_names[i], weight=Adj_M[i, j])

    # Add confounder nodes
    G.add_nodes_from(confounder_names)

    # Add edges from Adj_N (confounder-to-feature connections)
    num_confounders = Adj_N.shape[1]
    for j in range(num_confounders):
        for i in range(num_features):
            if Adj_N[i, j] != 0:
                G.add_edge(confounder_names[j], feature_names[i], weight=Adj_N[i, j])

    # Choose graph layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError("Unsupported layout. Choose 'spring', 'circular', or 'kamada_kawai'.")

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowstyle='-|>', arrowsize=20,
                           connectionstyle='arc3,rad=0.2')
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

    # Add edge labels for weights
    if edge_labels:
        edge_weights = nx.get_edge_attributes(G, 'weight')
        formatted_weights = {k: f"{v:.2f}" for k, v in edge_weights.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_weights, font_size=10)

    # save_dir = f"{path}/{fairness_metrics_dic_names[metric]}"
    if not os.path.exists(path):
        os.makedirs(path)  # Create directory if it does not exist
    save_path = f'{path}/{name}.png'

    plt.savefig(save_path)
    plt.clf()  # Clear the plot for future use
    plt.close()


def generate_discretization_no_d(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)
    Ux1 = np.random.normal(0, 1, n_samples)
    Ux2 = np.random.normal(0, 1, n_samples)
    Covariance_matrix = np.array([[1, 0.5], [0.5, 1]])
    # [X1, X2]^T = Covariance_matrix^{1/2} * [Ux1, Ux2]^T
    X = np.dot(np.linalg.cholesky(Covariance_matrix), np.array([Ux1, Ux2]))
    X1 = X[0, :]
    X2 = X[1, :]

    Y = np.random.binomial(1, 0.5, n_samples)


    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (10, 10, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_discretization_1(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility
    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)
    Ux1 = np.random.normal(0, 1, n_samples)
    Ux2 = np.random.normal(0, 1, n_samples)
    Covariance_matrix = np.array([[1, 0.5], [0.5, 1]])
    # [X1, X2]^T = Covariance_matrix^{1/2} * [Ux1, Ux2]^T
    X = np.dot(np.linalg.cholesky(Covariance_matrix), np.array([Ux1, Ux2]))
    X1 = X[0, :]
    X2 = X[1, :]

    Y = np.random.binomial(1, 0.5, n_samples)


    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'A': A,
        'Y': Y
    })
    df['X1'] = pd.cut(df['X1'], bins=10, labels=False)
    df['X2'] = pd.cut(df['X2'], bins=10, labels=False)
    features = ['X1', 'X2']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (10, 10, 2, 2)
    return df, features, sensitive_attribute, output, bins

def generate_discretization_2(n_samples=1000000, parameters=[], seed=0):
    np.random.seed(seed)  # For reproducibility

    # Generate A (binary sensitive attribute)
    A = np.random.binomial(1, 0.5, n_samples)
    Ux1 = np.random.normal(0, 1, n_samples)
    Ux2 = np.random.normal(0, 1, n_samples)  # Changed to 0.5 probability

    Covariance_matrix = np.array([[1, 0.5], [0.5, 1]])

    # [X1, X2]^T = Covariance_matrix^{1/2} * [Ux1, Ux2]^T
    X = np.dot(np.linalg.cholesky(Covariance_matrix), np.array([Ux1, Ux2]))
    X1 = X[0, :]
    X2 = X[1, :]

    Y = np.random.binomial(1, 0.5, n_samples)

    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'A': A,
        'Y': Y
    })

    # quantiles_X1 = np.percentile(df['X1'], np.linspace(0, 100, 20))  # 21 edges for 20 bins
    # quantiles_X2 = np.percentile(df['X2'], np.linspace(0, 100, 20))
    #
    # df['X1'] = np.digitize(df['X1'], bins=quantiles_X1, right=True)
    # df['X2'] = np.digitize(df['X2'], bins=quantiles_X2, right=True)
    df['X1'], kk = pd.qcut(df['X1'], q=10, labels=False, retbins=True, duplicates='drop')
    df['X2'], kk = pd.qcut(df['X2'], q=10, labels=False, retbins=True, duplicates='drop')
    features = ['X1', 'X2']
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (10, 10, 2, 2)

    return df, features, sensitive_attribute, output, bins


def unified_model_corrected_confounders_dis(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier = True, number_of_neurons = 5):

    # nubmer of features = 6
    np.random.seed(seed)  # For reproducibility
    uy = np.random.normal(0, 1, n_samples)
    u1 = np.random.normal(0, 1, n_samples)
    u2 = np.random.normal(0, 1, n_samples)
    u3 = np.random.normal(0, 1, n_samples)
    u4 = np.random.normal(0, 1, n_samples)
    a = np.random.binomial(1, p_a, n_samples)#
    # a is zeros
    # a = np.zeros(n_samples)


    # uniform distribution, Gaussian distribution, Gamma distribution
    Z2 = np.random.uniform(0, 2, n_samples)
    Z1 = np.random.gamma(2, 1, n_samples)

    # Identity matrix
    I_6 = np.eye(6)

    # Construct U and Z
    U = np.column_stack([uy, u1, u2, u3, u4,a])  # Shape: (n_samples, 7)
    Z = np.column_stack([Z1, Z2])  # Shape: (n_samples, 3)

    # Compute V
    I_minus_Adj_M = I_6 - Adj_M
    V = np.linalg.inv(I_minus_Adj_M) @ U.T + np.linalg.inv(I_minus_Adj_M) @ Adj_N @ Z.T  # Shape: (7, n_samples)

    # Extract variables
    Y, X1, X2, X3, X4, A = V
    Y = (Y-np.mean(Y))/np.std(Y)
    Y = np.where(Y > thy, 1, 0)
    # A = np.random.binomial(1, p_a, n_samples)
    df0 = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y})
    df1 = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    features = ['X1', 'X2', 'X3', 'X4']
    sensitive_attribute = ['A']
    output = ['Y']
    for feature in features:
        df1[feature] = pd.cut(df1[feature], bins=parameters[0], labels=False)
    bins = (parameters[0], parameters[0], parameters[0], parameters[0], 2, 2)

    df2 = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })
    for feature in features:
        df2[feature],kk = pd.qcut(df2[feature], q=parameters[0], labels=False, retbins=True, duplicates='drop')
    # return df
    # quantiles_X1 = np.percentile(df2['X1'], np.linspace(0, 100, parameters[0] ))  # 21 edges for 20 bins
    # quantiles_X2 = np.percentile(df2['X2'], np.linspace(0, 100, parameters[0] ))
    # quantiles_X3 = np.percentile(df2['X3'], np.linspace(0, 100, parameters[0] ))
    # quantiles_X4 = np.percentile(df2['X4'], np.linspace(0, 100, parameters[0] ))
    # df2['X1'] = np.digitize(df2['X1'], bins=quantiles_X1, right=False)
    # df2['X2'] = np.digitize(df2['X2'], bins=quantiles_X2, right=False)
    # df2['X3'] = np.digitize(df2['X3'], bins=quantiles_X3, right=False)
    # df2['X4'] = np.digitize(df2['X4'], bins=quantiles_X4, right=False)
    # print(df2['X1'].value_counts())

    df3 = df0.copy()
    features = ['X1', 'X2', 'X3', 'X4']

    # Apply PCA to preserve dependencies
    pca = PCA(n_components=len(features))
    X_pca = pca.fit_transform(df3[features])

    # Discretize PCA components to find bin edges
    discretizer = KBinsDiscretizer(n_bins=parameters[0], encode='ordinal', strategy='quantile')
    X_discrete = discretizer.fit_transform(X_pca)

    # Get the bin edges from the PCA-transformed space
    bin_edges_pca = discretizer.bin_edges_

    # Map PCA bin edges back to the original feature space
    mapped_bin_edges = pca.inverse_transform(np.array([edges for edges in bin_edges_pca]).T)

    # print(mapped_bin_edges)
    # Use mapped bins to discretize the original features
    for i, feature in enumerate(features):
        print(mapped_bin_edges[:, i])
        df3[feature] = np.digitize(df3[feature], bins=mapped_bin_edges[:, i], right=False)



    return df0, df1, df2, df3, features, sensitive_attribute, output, bins


def unified_model_corrected_confounders_dis2(Adj_M, Adj_N, p_a, thy, n_samples=1000000, parameters=[], seed=0, LR_classifier=True, number_of_neurons=5):

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate latent variables
    uy = np.random.normal(0, 1, n_samples)
    u1 = np.random.normal(0, 1, n_samples)
    u2 = np.random.normal(0, 1, n_samples)
    u3 = np.random.normal(0, 1, n_samples)
    u4 = np.random.normal(0, 1, n_samples)
    a = np.random.binomial(1, p_a, n_samples)  # Binary sensitive attribute

    # Continuous variables
    Z2 = np.random.uniform(0, 2, n_samples)
    Z1 = np.random.gamma(2, 1, n_samples)

    # Identity matrix
    I_6 = np.eye(6)

    # Construct U and Z
    U = np.column_stack([uy, u1, u2, u3, u4, a])  # Shape: (n_samples, 6)
    Z = np.column_stack([Z1, Z2])  # Shape: (n_samples, 2)

    # Compute V using inverse transformation
    I_minus_Adj_M = I_6 - Adj_M
    V = np.linalg.inv(I_minus_Adj_M) @ U.T + np.linalg.inv(I_minus_Adj_M) @ Adj_N @ Z.T  # Shape: (6, n_samples)

    # Extract variables
    Y, X1, X2, X3, X4, A = V
    Y = (Y - np.mean(Y)) / np.std(Y)
    Y = np.where(Y > thy, 1, 0)

    # Create the original dataset (df0)
    df0 = pd.DataFrame({
        'X1': X1 / np.std(X1),
        'X2': X2 / np.std(X2),
        'X3': X3 / np.std(X3),
        'X4': X4 / np.std(X4),
        'A': A,
        'Y': Y
    })

    # Discretized dataset (df1) using equal-width binning
    df1 = df0.copy()
    for feature in ['X1', 'X2', 'X3', 'X4']:
        df1[feature] = pd.cut(df1[feature], bins=parameters[0], labels=False)

    # Discretized dataset (df2) using quantile binning
    df2 = df0.copy()
    for feature in ['X1', 'X2', 'X3', 'X4']:
        df2[feature], _ = pd.qcut(df2[feature], q=parameters[0], labels=False, retbins=True, duplicates='drop')

    # --- K-Means in PCA Projection Space for Adaptive Binning ---
    df3 = df0.copy()
    features = ['X1', 'X2', 'X3', 'X4']

    # Apply PCA to preserve dependencies
    pca = PCA(n_components=len(features))
    X_pca = pca.fit_transform(df3[features])

    # Apply K-Means to identify clusters (bins) in PCA space
    kmeans = KMeans(n_clusters=parameters[0], random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)

    # Get cluster centroids in PCA space
    cluster_centroids_pca = kmeans.cluster_centers_

    # Map cluster centroids back to the original feature space
    mapped_bin_edges = pca.inverse_transform(cluster_centroids_pca)

    # Use mapped centroids to discretize original features
    for i, feature in enumerate(features):
        edges = mapped_bin_edges[:, i]
        # sort the bin edges
        edges = np.sort(edges)
        df3[feature] = np.digitize(df3[feature], bins=edges, right=False)

    # Define metadata for feature sets
    sensitive_attribute = ['A']
    output = ['Y']
    bins = (parameters[0], parameters[0], parameters[0], parameters[0], 2, 2)

    df4 = df0.copy()
    features = ['X1', 'X2', 'X3', 'X4']

    # Apply PCA to preserve dependencies
    pca = PCA(n_components=len(features))
    X_pca = pca.fit_transform(df4[features])

    # Discretize PCA components to find bin edges
    discretizer = KBinsDiscretizer(n_bins=parameters[0], encode='ordinal', strategy='quantile')
    X_discrete = discretizer.fit_transform(X_pca)

    # Get the bin edges from the PCA-transformed space
    bin_edges_pca = discretizer.bin_edges_

    # Map PCA bin edges back to the original feature space
    mapped_bin_edges = pca.inverse_transform(np.array([edges for edges in bin_edges_pca]).T)

    # Use mapped bins to discretize the original features
    for i, feature in enumerate(features):
        df4[feature] = np.digitize(df4[feature], bins=mapped_bin_edges[:, i], right=False)

    return df0, df1, df2, df3, df4, features, sensitive_attribute, output, bins