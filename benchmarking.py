
import numpy as np
import pandas as pd

from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage, ACSMobility, ACSTravelTime
from sklearn.metrics import normalized_mutual_info_score
from category_encoders import TargetEncoder
from folktables import generate_categories
from folktables import BasicProblem
from folktables import acs

# THis code us used to test the FeaturesQuantifier class with the Synthetic Dataset from paper "Khodadadian, Sajad, et al. "Information theoretic measures for fairness-aware feature selection." arXiv preprint arXiv:2106.00772 (2021)."
import numpy as np
import pandas as pd


import math
import dit
import dit.pid
import matplotlib.pyplot as plt
from dit.npdist import Distribution
from dit.pid.measures.ibroja import BROJAOptimizer
from dit.multivariate import coinformation
from dit.shannon import mutual_information as I
from dit.shannon import entropy as H
from sklearn.model_selection import train_test_split

from dit.pid.measures import PID_BROJA
from dit.algorithms import pid_broja
from dit.divergences import variational_distance, kullback_leibler_divergence

from sklearn.metrics.pairwise import rbf_kernel

from typing import Literal
import os
import matplotlib.ticker as ticker
import itertools
from sklearn.metrics import log_loss, accuracy_score
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
#
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
import inspect
from src.shaplyze import ShaplyzeEstimator
# from src.datasets import * # for the datasets with EFQ
from src.datasets_samples import * # for datasets with samples

from src.measures import *

from src.midist import MIDist as I_dis
from src.migaussian import MIGaussian as I_gu
from src.classifier import ClassifierBase as ClassifierBase
import argparse

parser = argparse.ArgumentParser(description="A program that responds to terminal directives.")
# Add arguments
parser.add_argument('--ID', type=int, help='range of the models to be tested')
args = parser.parse_args()
ID = args.ID

# id_list = [535, 536, 537, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 335, 341, 329, 150, 138, 144]
# ID = id_list[ID] if ID < len(id_list) else 5000


k = 3

models_list = [generate_dataset_folktables_RAC1P_WB_selective_updated,
               generate_dataset_folktables_PublicCoverage_selective_2_updated,
               generate_dataset_Employment_selective_updated,
               generate_data_model_ProPublicaCOMPASDataset_enhanced_preprocessing_updated,
               CensusIncomeKDD_updated,
               adult_data_updated,
               Heritage_updated]
N_list = [200, 50, 200, 100, 90, 110, 170]
seeds_list = [i for i in range(5)]  # Seeds for the classifier-based measures

measures = {
    "I(A;X_S)": {'constructor': I_Xs_A, 'bins': True, 'surrogate':[None,SurrogateMeasure], 'seeds': [None, seeds_list]},
    "I(Y;X_S|A)": {'constructor': I_Y_Xs_given_AXsc, 'bins': True, 'surrogate':[None,SurrogateMeasure], 'seeds': [None, seeds_list]},
    "SI(A;X_S,Y)": {'constructor': SI_A_Xs_Y, 'bins': True, 'surrogate':[None,SurrogateMeasure], 'seeds': [None, seeds_list]},
    "SI(X_S;A,Y)": {'constructor': SI_Xs_A_Y, 'bins': True, 'surrogate':[None,SurrogateMeasure], 'seeds': [None, seeds_list]},
    "I(A;X_S|Y)": {'constructor': I_Xs_A_given_Y, 'bins': True, 'surrogate':[None,SurrogateMeasure], 'seeds': [None, seeds_list]},
    "I(A;X_S)I(A;X_S|Y)SI(Y;X_S,A)": {'constructor': I_A_Xs_times_IAXs_given_Y_times_SI_Y_Xs_A, 'bins': True, 'surrogate':[None,SurrogateMeasure], 'seeds': [None, seeds_list]},
    "I(A;X_S)I(A;X_S|Y)": {'constructor': I_A_Xs_times_IAXs_given_Y, 'bins': True, 'surrogate':[None,SurrogateMeasure], 'seeds': [None, seeds_list]},
    "V^2(X_S;A)": {'constructor': DisCov_Xs_A, 'bins': True, 'surrogate':[None,SurrogateMeasure], 'seeds': [None, seeds_list]},
    "V^2(X_S;Y)":{ 'constructor': DisCov_Xs_Y, 'bins': True, 'surrogate':[None,SurrogateMeasure], 'seeds': [None, seeds_list]},
    "V^2(X_S;A|Y)": {'constructor': DisCov_Xs_A_given_Y, 'bins': True, 'surrogate':[None,SurrogateMeasure], 'seeds': [None, seeds_list]},
    "I_g(A;X_S)": {'constructor': Ig_Xs_A, 'bins': False, 'surrogate':[None, SurrogateMeasure], 'seeds': [None, seeds_list]},
    "I_g(Y;X_S|A)": {'constructor': Ig_Y_Xs_given_AXsc, 'bins': False, 'surrogate':[None, SurrogateMeasure], 'seeds': [None, seeds_list]},
    "SI_g(A;X_S,Y)": {'constructor': SIg_A_Xs_Y, 'bins': False, 'surrogate':[None, SurrogateMeasure], 'seeds': [None, seeds_list]},
    "SI_g(X_S;A,Y)": {'constructor': SIg_Xs_A_Y, 'bins': False, 'surrogate':[None, SurrogateMeasure], 'seeds': [None, seeds_list]},
    "I_g(A;X_S|Y)": {'constructor': Ig_Xs_A_given_Y, 'bins': False, 'surrogate':[None, SurrogateMeasure], 'seeds': [None, seeds_list]},
    "I_g(A;X_S)I_g(A;X_S|Y)SI_g(Y;X_S,A)": {'constructor': Ig_A_Xs_times_IAXs_given_Y_times_SI_Y_Xs_A, 'bins': False, 'surrogate':[None, SurrogateMeasure], 'seeds': [None, seeds_list]},
    "I_g(A;X_S)I_g(A;X_S|Y)": {'constructor': Ig_A_Xs_times_IAXs_given_Y, 'bins': False, 'surrogate':[None, SurrogateMeasure], 'seeds': [None, seeds_list]},
    "HSIC(X_S;Y)": {'constructor': HSIC_Xs_Y, 'bins': False, 'surrogate':[None, SurrogateMeasure], 'seeds': [None, seeds_list]},
    "NOCCO(X_S;Y)": {'constructor': NOCCO_Xs_Y, 'bins': False, 'surrogate':[None, SurrogateMeasure], 'seeds': [None, seeds_list]},
    "HSIC(X_S;A)": {'constructor': HSIC_Xs_A, 'bins': False, 'surrogate':[None, SurrogateMeasure], 'seeds': [None, seeds_list]},
    "NOCCO(X_S;A)": {'constructor': NOCCO_Xs_A, 'bins': False, 'surrogate':[None, SurrogateMeasure], 'seeds': [None, seeds_list]},
    "HSIC(X_S;A|Y)": {'constructor': HSIC_Xs_A_given_Y, 'bins': False, 'surrogate':[None, SurrogateMeasure], 'seeds': [None, seeds_list]},
    "NOCCO(X_S;A|Y)": {'constructor': NOCCO_Xs_A_given_Y, 'bins': False, 'surrogate':[None, SurrogateMeasure], 'seeds': [None, seeds_list]},
    "MMD(X_S;Y)": {'constructor': MMD_Xs_Y, 'bins': False, 'surrogate':[None, SurrogateMeasure], 'seeds': [None, seeds_list]},
    "MMD(X_S;A)": {'constructor': MMD_Xs_A, 'bins': False, 'surrogate':[None, SurrogateMeasure], 'seeds': [None, seeds_list]},
    "MMD(X_S;A|Y)": {'constructor': MMD_Xs_A_given_Y, 'bins': False, 'surrogate':[None, SurrogateMeasure], 'seeds': [None, seeds_list]},
    "SI(A;X_S,hat Y)": {'constructor': SI_A_Xs_Y, 'bins': True, 'surrogate':[SurrogateMeasure_Y_by_hat_Y, SurrogateMeasure_Y_by_hat_Y_Xs_hat_YS], 'seeds': [seeds_list, seeds_list]},
    "SI_g(A;X_S,hat Y)":{ 'constructor': SIg_A_Xs_Y, 'bins': False, 'surrogate':[SurrogateMeasure_Y_by_hat_Y, SurrogateMeasure_Y_by_hat_Y_Xs_hat_YS], 'seeds': [seeds_list, seeds_list]},
    "Accuracy": {'constructor': AccuracyMeasure, 'bins': False, 'surrogate':[None], 'seeds': [seeds_list]},
    "DP": {'constructor': DPMeasure, 'bins': False, 'surrogate':[None], 'seeds': [seeds_list]},
    "EO": {'constructor': EOMeasure, 'bins': False, 'surrogate':[None], 'seeds': [seeds_list]},
}

it = 0

for k in range(len(models_list)):
    for key in measures.keys():
        constructor = measures[key]['constructor']
        bins_exits = measures[key]['bins']
        surrogate_list = measures[key]['surrogate']
        experiment_seeds = measures[key]['seeds']
        for i in range(len(surrogate_list)):

            seeds = experiment_seeds[i]
            surrogate = surrogate_list[i]
            there_is_seeds = True
            if seeds is None:
                seeds = [0]
                there_is_seeds = False
            for s in seeds:
                if it not in [ID]:
                    it += 1
                    continue
                print(it)
                print("Running measure:", key, "on model:", models_list[k].__name__, "with N =", N_list[k], "and seed =", s)
                N = N_list[k]
                data_model = models_list[k]
                df, features, sensitive_attribute, output, bins, categorical_features = data_model()
                X, A, Y = features, sensitive_attribute, output
                clf = MLPClassifier(hidden_layer_sizes=(N,), max_iter=1000, random_state=s)
                kwargs = dict(
                    mode='replace',
                    classifier=clf,
                    categorical_features=categorical_features,
                    split_ratio=0.33,
                    seed=s
                )
                if bins_exits:
                    kwargs['bins'] = bins
                # Construct measure
                if surrogate is None and there_is_seeds:
                    measure = constructor(df = df, X = X, A = A, Y = Y, **kwargs)
                elif surrogate is not None and there_is_seeds:
                    measure = surrogate(df = df, X = X, A = A, Y = Y, measure_class= constructor, **kwargs)
                else:
                    measure = constructor(df = df, X = X, A = A, Y = Y, bins=bins) if bins_exits else constructor(df, X, A, Y)
                shapley_estimator = ShaplyzeEstimator(measure)
                all_shap_values = shapley_estimator.get_sh_values()
                print(it, "Measure:", measure.name())
                print("Shapley values:", all_shap_values)
                # save the results
                folder = "results/"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                file_name = f'Shapley_values_{it}.npy'
                file_path = os.path.join(folder, file_name)
                np.save(file_path, all_shap_values)
                it += 1




