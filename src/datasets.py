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
