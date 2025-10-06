import numpy as np
from folktables import ACSIncome
from src.shaplyze import ShaplyzeEstimator

d1 = 0.1

seeds_list = [i for i in range(5)]  # Seeds for the classifier-based measures
# measures = {
#     "I(A;X_S)": {'Baseline': ['SP'], 'bins': True, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "I(Y;X_S|A,X_{S^C})": {'Baseline': ['A'], 'bins': True, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "SI(A;X_S,Y)": {'Baseline': ['SP'], 'bins': True, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "SI(X_S;A,Y)": {'Baseline': ['SP'], 'bins': True, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "I(A;X_S|Y)": {'Baseline': ['EO'], 'bins': True, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "I(A;X_S)I(A;X_S|Y)SI(Y;X_S,A)": {'Baseline': ['SP','EO'], 'bins': True, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "I(A;X_S)I(A;X_S|Y)": {'Baseline': ['SP','EO'], 'bins': True, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "V^2(X_S,A)": {'Baseline': ['SP'], 'bins': True, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "V^2(X_S,Y)":{ 'Baseline': ['A'], 'bins': True, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "V^2_Y(X_S,A)": {'Baseline': ['EO'], 'bins': True, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "I_g(A;X_S)": {'Baseline': ['SP'], 'bins': False, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "I_g(Y;X_S|A,X_{S^C})": {'Baseline': ['A'], 'bins': False, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "SI_g(A;X_S,Y)": {'Baseline': ['SP'], 'bins': False, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "SI_g(X_S;A,Y)": {'Baseline': ['SP'], 'bins': False, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "I_g(A;X_S|Y)": {'Baseline': ['EO'], 'bins': False, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "I_g(A;X_S)I_g(A;X_S|Y)SI_g(Y;X_S,A)": {'Baseline': ['SP', 'EO'], 'bins': False, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "I_g(A;X_S)I_g(A;X_S|Y)": {'Baseline': ['SP', 'EO'], 'bins': False, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "HSIC(X_S,Y)": {'Baseline': ['A'], 'bins': False, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "NOCCO(X_S,Y)": {'Baseline': ['A'], 'bins': False, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "HSIC(X_S,A)": {'Baseline': ['SP'], 'bins': False, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "NOCCO(X_S,A)": {'Baseline': ['SP'], 'bins': False, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "HSIC_Y(X_S,A)": {'Baseline': ['EO'], 'bins': False, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "NOCCO_Y(X_S,A)": {'Baseline': ['EO'], 'bins': False, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "MMD^2(X_{S,Y=0},X_{S,Y=1})": {'Baseline': ['A'], 'bins': False, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "MMD^2(X_{S,A=0},X_{S,A=1})": {'Baseline': ['SP'], 'bins': False, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "MMD^2_Y(X_{S,A=0},X_{S,A=1})": {'Baseline': ['EO'], 'bins': False, 'surrogate':[False,True], 'seeds': [None, seeds_list], 'results':[[], []]},
#     "SI(A;X_S,hat Y)": {'Baseline': ['SP'], 'bins': True, 'surrogate':[False,True], 'seeds': [seeds_list, seeds_list], 'results':[[], []]},
#     "SI_g(A;X_S,hat Y)":{ 'Baseline': ['SP'], 'bins': False, 'surrogate':[False,True], 'seeds': [seeds_list, seeds_list], 'results':[[], []]},
#     "Accuracy": {'Baseline': ['A'], 'bins': False, 'surrogate':[False], 'seeds': [seeds_list], 'results':[[]]},
#     "SP": {'Baseline': ['SP'], 'bins': False, 'surrogate':[False], 'seeds': [seeds_list], 'results':[[]]},
#     "EO": {'Baseline': ['EO'], 'bins': False, 'surrogate':[False], 'seeds': [seeds_list], 'results':[[]]},
# }
with_access = []
without_access = []
target = 'EO'
use_causal_info = False
datasets = ["ACSIncome",
            "ACSCoverage",
            "ACSEMployment",
            "COMPAS",
            "CensusIncomeKDD",
            "adult",
            "Health"]
features_lists = [
    ['COW', 'SEX', 'WKHP', 'SCHL'],
    ['AGEP', 'ESP', 'SEX', 'MIG', 'FER'],
    ['MIG', 'MAR', 'ESP', 'RAC1P', 'AGEP'],
    ['age_cat', 'c_charge_degree', 'sex', 'priors_count', 'Length Of Stay'],
    ['education', 'marital_stat', 'race', 'capital_gains', 'capital_losses', 'num_emp'],
    ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'relationship'],
    ['Year', 'PayDelay', 'PlaceSvc', 'DSFS', 'Sex'],
]
causal_info_lists = [[['SEX', 'SCHL']],
                     [['AGEP', 'ESP']],
                     [['MAR', 'MIG']],
                     [['sex', 'priors_count'], ['age_cat', 'c_charge_degree'], ['age_cat', 'priors_count']],
                     [['race', 'education'], ['capital_gains', 'marital_stat']],
                     [['age', 'educational-num']],
                     [['Sex','PlaceSvc']]
                     ]
order_over_datasets = [0, 2, 1, 5, 4, 3, 6]  # Order of datasets in the resultsclea
acc_baseline_ind = [176, 177, 178, 179, 180]  # Index of the accuracy baseline in the results
SP_baseline_ind = [181, 182, 183, 184, 185]  # Index of the SP baseline in the results
EO_baseline_ind = [186, 187, 188, 189, 190]  # Index of the EO baseline in the results
baseline_indices = {"A": acc_baseline_ind, "SP": SP_baseline_ind, "EO": EO_baseline_ind}
# order_over_datasets = [0]  # Order of datasets in the results
data_set_list = []
measure_list = []
baseline_list = []
with_access_list = []
with_causal_info_list = []
error_list = []
error_std_list = []
for ind_dataset in order_over_datasets:
    print(f"Dataset: {datasets[ind_dataset]}")
    causal_info = causal_info_lists[ind_dataset]

    measures = {
        "$v_{SP,MI}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                     'results': [[], []]},
        "$v_{A,CMI}$": {'Baseline': ['A'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                               'results': [[], []]},
        "$v_{SP,SI-A}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                        'results': [[], []]},
        "$v_{SP,SI-X_S}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                        'results': [[], []]},
        "$v_{EO,CMI}$": {'Baseline': ['EO'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                       'results': [[], []]},
        "$v_{D,MI.CMI.SI}$": {'Baseline': ['SP', 'EO'], 'bins': True, 'surrogate': [False, True],
                      'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{D,MI.CMI}$": {'Baseline': ['SP', 'EO'], 'bins': True, 'surrogate': [False, True],
                   'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{SP,DC}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                       'results': [[], []]},
        "$v_{A,DC}$": {'Baseline': ['A'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                       'results': [[], []]},
        "$v_{EO,CDC}$": {'Baseline': ['EO'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                         'results': [[], []]},
        "$v_{SP,MI_g}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                       'results': [[], []]},
        "$v_{A,CMI_g}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                                 'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{SP,SI_g-A}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                          'results': [[], []]},
        "$v_{SP,SI_g-X_S}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                          'results': [[], []]},
        "$v_{EO,CMI_g}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                         'results': [[], []]},
        "$v_{D,MI_g.CMI_g.SI_g}$": {'Baseline': ['SP', 'EO'], 'bins': False, 'surrogate': [False, True],
                            'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{D,MI_g.CMI_g}$": {'Baseline': ['SP', 'EO'], 'bins': False, 'surrogate': [False, True],
                       'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{A,HSIC}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                                'seeds': [None, seeds_list],
                                'results': [[], []]},
        "$v_{A,NOCCO}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                                 'seeds': [None, seeds_list],
                                 'results': [[], []]},
        "$v_{SP,HSIC}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                                'seeds': [None, seeds_list],
                                'results': [[], []]},
        "$v_{SP,NOCCO}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                                 'seeds': [None, seeds_list],
                                 'results': [[], []]},
        "$v_{EO,CHSIC}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True],
                                   'seeds': [None, seeds_list],
                                   'results': [[], []]},
        "$v_{EO,CNOCCO}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True],
                                    'seeds': [None, seeds_list],
                                    'results': [[], []]},
        "$v_{A,MMD}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                                               'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{SP,MMD}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                                               'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{EO,CMMD}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True],
                                                  'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{SP,SI-A,\hat Y}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True],
                            'seeds': [seeds_list, seeds_list], 'results': [[], []]},
        "$v_{SP,SI_g-A,\hat Y}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                              'seeds': [seeds_list, seeds_list], 'results': [[], []]},
        "$v_{A,Acc}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [True], 'seeds': [seeds_list],
                           'results': [[]]},
        "$v_{SP,SPD}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [True], 'seeds': [seeds_list],
                          'results': [[]]},
        "$v_{EO,EOD}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [True], 'seeds': [seeds_list],
                          'results': [[]]},
    }

    measures_std = {
        "$v_{SP,MI}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                        'results': [[], []]},
        "$v_{A,CMI}$": {'Baseline': ['A'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                        'results': [[], []]},
        "$v_{SP,SI-A}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                          'results': [[], []]},
        "$v_{SP,SI-X_S}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                            'results': [[], []]},
        "$v_{EO,CMI}$": {'Baseline': ['EO'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                         'results': [[], []]},
        "$v_{D,MI.CMI.SI}$": {'Baseline': ['SP', 'EO'], 'bins': True, 'surrogate': [False, True],
                              'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{D,MI.CMI}$": {'Baseline': ['SP', 'EO'], 'bins': True, 'surrogate': [False, True],
                           'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{SP,DC}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                        'results': [[], []]},
        "$v_{A,DC}$": {'Baseline': ['A'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                       'results': [[], []]},
        "$v_{EO,CDC}$": {'Baseline': ['EO'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                         'results': [[], []]},
        "$v_{SP,MI_g}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                          'results': [[], []]},
        "$v_{A,CMI_g}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                          'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{SP,SI_g-A}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                            'results': [[], []]},
        "$v_{SP,SI_g-X_S}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                              'seeds': [None, seeds_list],
                              'results': [[], []]},
        "$v_{EO,CMI_g}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                           'results': [[], []]},
        "$v_{D,MI_g.CMI_g.SI_g}$": {'Baseline': ['SP', 'EO'], 'bins': False, 'surrogate': [False, True],
                                    'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{D,MI_g.CMI_g}$": {'Baseline': ['SP', 'EO'], 'bins': False, 'surrogate': [False, True],
                               'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{A,HSIC}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                         'seeds': [None, seeds_list],
                         'results': [[], []]},
        "$v_{A,NOCCO}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                          'seeds': [None, seeds_list],
                          'results': [[], []]},
        "$v_{SP,HSIC}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                          'seeds': [None, seeds_list],
                          'results': [[], []]},
        "$v_{SP,NOCCO}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                           'seeds': [None, seeds_list],
                           'results': [[], []]},
        "$v_{EO,CHSIC}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True],
                           'seeds': [None, seeds_list],
                           'results': [[], []]},
        "$v_{EO,CNOCCO}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True],
                            'seeds': [None, seeds_list],
                            'results': [[], []]},
        "$v_{A,MMD}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                        'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{SP,MMD}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                         'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{EO,CMMD}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True],
                          'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{SP,SI-A,\hat Y}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True],
                                 'seeds': [seeds_list, seeds_list], 'results': [[], []]},
        "$v_{SP,SI_g-A,\hat Y}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                                   'seeds': [seeds_list, seeds_list], 'results': [[], []]},
        "$v_{A,Acc}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [True], 'seeds': [seeds_list],
                        'results': [[]]},
        "$v_{SP,SPD}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [True], 'seeds': [seeds_list],
                         'results': [[]]},
        "$v_{EO,EOD}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [True], 'seeds': [seeds_list],
                         'results': [[]]},
    }

    which_dataset = ind_dataset

    per_dataset = 191

    total = per_dataset
    it = per_dataset * which_dataset
    previous_sh = 0
    for key in measures.keys():
        # print('Measure:', key)
        for r in range(len(measures[key]['results'])):
            # print('Result:', measures[key]['results'][r])
            if measures[key]['seeds'][r] is None:
                # print('Seed:', measures[key]['seeds'][r])
                base = per_dataset * which_dataset
                for b_it in range(len(measures[key]['Baseline'])):
                    baseline = measures[key]['Baseline'][b_it]
                    b_indices = [baseline_indices[baseline][l] + base for l in range(len(baseline_indices[baseline]))]
                    error = 0
                    error_local_list = []
                    for ind in b_indices:
                        # print('ind:', ind)
                        shapley_est = ShaplyzeEstimator(X=features_lists[ind_dataset], causal_structure_info=causal_info if use_causal_info else [])
                        error += shapley_est.get_error(ID=it,folder_path='droup_out_res/dictionary_res_06_26', baseline_dic_path=f'droup_out_res/dictionary_res_06_26/dic_{ind}.npy')
                        error_local_list.append(shapley_est.get_error(ID=it,folder_path='droup_out_res/dictionary_res_06_26', baseline_dic_path=f'droup_out_res/dictionary_res_06_26/dic_{ind}.npy'))
                    error = error / len(b_indices)
                    measures[key]['results'][r].append(error)
                    measures_std[key]['results'][r].append(np.std(error_local_list))
                    # print('key:', key, 'baseline:', baseline, 'error:', error, 'no access')
                # print('it:', it)
                it = it + 1

            else:
                base = per_dataset * which_dataset
                current_it = it
                for b_it in range(len(measures[key]['Baseline'])):
                    baseline = measures[key]['Baseline'][b_it]
                    b_indices = [baseline_indices[baseline][l] + base for l in range(len(baseline_indices[baseline]))]
                    error = 0
                    error_local_list = []
                    it = current_it
                    for ind in b_indices:
                        shapley_est = ShaplyzeEstimator(X=features_lists[ind_dataset],
                                                        causal_structure_info=causal_info if use_causal_info else [])
                        error += shapley_est.get_error(ID=it, folder_path=f'droup_out_res/dictionary_res_06_26',
                                                       baseline_dic_path=f'droup_out_res/dictionary_res_06_26/dic_{ind}.npy')
                        error_local_list.append(shapley_est.get_error(ID=it, folder_path=f'droup_out_res/dictionary_res_06_26',
                                                       baseline_dic_path=f'droup_out_res/dictionary_res_06_26/dic_{ind}.npy'))
                        it = it + 1
                    error = error / len(b_indices)
                    measures[key]['results'][r].append(error)
                    measures_std[key]['results'][r].append(np.std(error_local_list))
                    # print('key:', key, 'baseline:', baseline, 'error:', error, 'with access')
                    it = current_it + 5


    for key in measures.keys():
        for r in range(len(measures[key]['results'])):
            for b_it in range(len(measures[key]['Baseline'])):
                data_set_list.append(datasets[ind_dataset])
                measure_list.append(key)
                baseline_list.append(measures[key]['Baseline'][b_it])
                with_causal_info_list.append("NO")
                if measures[key]['surrogate'][r]:
                    with_access_list.append("YES")
                else:
                    with_access_list.append("NO")
                error_list.append(measures[key]['results'][r][b_it])
                error_std_list.append(measures_std[key]['results'][r][b_it])


use_causal_info = True

for ind_dataset in order_over_datasets:
    print(f"Dataset: {datasets[ind_dataset]}")
    causal_info = causal_info_lists[ind_dataset]

    measures = {
        "$v_{SP,MI}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                        'results': [[], []]},
        "$v_{A,CMI}$": {'Baseline': ['A'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                        'results': [[], []]},
        "$v_{SP,SI-A}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                          'results': [[], []]},
        "$v_{SP,SI-X_S}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                            'results': [[], []]},
        "$v_{EO,CMI}$": {'Baseline': ['EO'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                         'results': [[], []]},
        "$v_{D,MI.CMI.SI}$": {'Baseline': ['SP', 'EO'], 'bins': True, 'surrogate': [False, True],
                              'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{D,MI.CMI}$": {'Baseline': ['SP', 'EO'], 'bins': True, 'surrogate': [False, True],
                           'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{SP,DC}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                        'results': [[], []]},
        "$v_{A,DC}$": {'Baseline': ['A'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                       'results': [[], []]},
        "$v_{EO,CDC}$": {'Baseline': ['EO'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                         'results': [[], []]},
        "$v_{SP,MI_g}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                          'results': [[], []]},
        "$v_{A,CMI_g}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                          'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{SP,SI_g-A}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                            'results': [[], []]},
        "$v_{SP,SI_g-X_S}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                              'seeds': [None, seeds_list],
                              'results': [[], []]},
        "$v_{EO,CMI_g}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                           'results': [[], []]},
        "$v_{D,MI_g.CMI_g.SI_g}$": {'Baseline': ['SP', 'EO'], 'bins': False, 'surrogate': [False, True],
                                    'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{D,MI_g.CMI_g}$": {'Baseline': ['SP', 'EO'], 'bins': False, 'surrogate': [False, True],
                               'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{A,HSIC}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                         'seeds': [None, seeds_list],
                         'results': [[], []]},
        "$v_{A,NOCCO}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                          'seeds': [None, seeds_list],
                          'results': [[], []]},
        "$v_{SP,HSIC}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                          'seeds': [None, seeds_list],
                          'results': [[], []]},
        "$v_{SP,NOCCO}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                           'seeds': [None, seeds_list],
                           'results': [[], []]},
        "$v_{EO,CHSIC}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True],
                           'seeds': [None, seeds_list],
                           'results': [[], []]},
        "$v_{EO,CNOCCO}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True],
                            'seeds': [None, seeds_list],
                            'results': [[], []]},
        "$v_{A,MMD}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                        'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{SP,MMD}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                         'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{EO,CMMD}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True],
                          'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{SP,SI-A,\hat Y}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True],
                                 'seeds': [seeds_list, seeds_list], 'results': [[], []]},
        "$v_{SP,SI_g-A,\hat Y}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                                   'seeds': [seeds_list, seeds_list], 'results': [[], []]},
        "$v_{A,Acc}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [True], 'seeds': [seeds_list],
                        'results': [[]]},
        "$v_{SP,SPD}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [True], 'seeds': [seeds_list],
                         'results': [[]]},
        "$v_{EO,EOD}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [True], 'seeds': [seeds_list],
                         'results': [[]]},
    }

    measures_std = {
        "$v_{SP,MI}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                        'results': [[], []]},
        "$v_{A,CMI}$": {'Baseline': ['A'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                        'results': [[], []]},
        "$v_{SP,SI-A}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                          'results': [[], []]},
        "$v_{SP,SI-X_S}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                            'results': [[], []]},
        "$v_{EO,CMI}$": {'Baseline': ['EO'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                         'results': [[], []]},
        "$v_{D,MI.CMI.SI}$": {'Baseline': ['SP', 'EO'], 'bins': True, 'surrogate': [False, True],
                              'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{D,MI.CMI}$": {'Baseline': ['SP', 'EO'], 'bins': True, 'surrogate': [False, True],
                           'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{SP,DC}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                        'results': [[], []]},
        "$v_{A,DC}$": {'Baseline': ['A'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                       'results': [[], []]},
        "$v_{EO,CDC}$": {'Baseline': ['EO'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                         'results': [[], []]},
        "$v_{SP,MI_g}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                          'results': [[], []]},
        "$v_{A,CMI_g}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                          'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{SP,SI_g-A}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                            'results': [[], []]},
        "$v_{SP,SI_g-X_S}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                              'seeds': [None, seeds_list],
                              'results': [[], []]},
        "$v_{EO,CMI_g}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                           'results': [[], []]},
        "$v_{D,MI_g.CMI_g.SI_g}$": {'Baseline': ['SP', 'EO'], 'bins': False, 'surrogate': [False, True],
                                    'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{D,MI_g.CMI_g}$": {'Baseline': ['SP', 'EO'], 'bins': False, 'surrogate': [False, True],
                               'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{A,HSIC}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                         'seeds': [None, seeds_list],
                         'results': [[], []]},
        "$v_{A,NOCCO}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                          'seeds': [None, seeds_list],
                          'results': [[], []]},
        "$v_{SP,HSIC}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                          'seeds': [None, seeds_list],
                          'results': [[], []]},
        "$v_{SP,NOCCO}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                           'seeds': [None, seeds_list],
                           'results': [[], []]},
        "$v_{EO,CHSIC}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True],
                           'seeds': [None, seeds_list],
                           'results': [[], []]},
        "$v_{EO,CNOCCO}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True],
                            'seeds': [None, seeds_list],
                            'results': [[], []]},
        "$v_{A,MMD}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                        'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{SP,MMD}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                         'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{EO,CMMD}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True],
                          'seeds': [None, seeds_list], 'results': [[], []]},
        "$v_{SP,SI-A,\hat Y}$": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True],
                                 'seeds': [seeds_list, seeds_list], 'results': [[], []]},
        "$v_{SP,SI_g-A,\hat Y}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                                   'seeds': [seeds_list, seeds_list], 'results': [[], []]},
        "$v_{A,Acc}$": {'Baseline': ['A'], 'bins': False, 'surrogate': [True], 'seeds': [seeds_list],
                        'results': [[]]},
        "$v_{SP,SPD}$": {'Baseline': ['SP'], 'bins': False, 'surrogate': [True], 'seeds': [seeds_list],
                         'results': [[]]},
        "$v_{EO,EOD}$": {'Baseline': ['EO'], 'bins': False, 'surrogate': [True], 'seeds': [seeds_list],
                         'results': [[]]},
    }

    which_dataset = ind_dataset

    per_dataset = 191

    total = per_dataset
    it = per_dataset * which_dataset
    previous_sh = 0
    for key in measures.keys():
        # print('Measure:', key)
        for r in range(len(measures[key]['results'])):
            # print('Result:', measures[key]['results'][r])
            if measures[key]['seeds'][r] is None:
                # print('Seed:', measures[key]['seeds'][r])
                base = per_dataset * which_dataset
                for b_it in range(len(measures[key]['Baseline'])):
                    baseline = measures[key]['Baseline'][b_it]
                    b_indices = [baseline_indices[baseline][l] + base for l in range(len(baseline_indices[baseline]))]
                    error = 0
                    error_local_list = []
                    for ind in b_indices:
                        # print('ind:', ind)
                        shapley_est = ShaplyzeEstimator(X=features_lists[ind_dataset], causal_structure_info=causal_info if use_causal_info else [])
                        error += shapley_est.get_error(ID=it,folder_path='droup_out_res/dictionary_res_06_26', baseline_dic_path=f'droup_out_res/dictionary_res_06_26/dic_{ind}.npy')
                        error_local_list.append(shapley_est.get_error(ID=it,folder_path='droup_out_res/dictionary_res_06_26', baseline_dic_path=f'droup_out_res/dictionary_res_06_26/dic_{ind}.npy'))
                    error = error / len(b_indices)
                    measures[key]['results'][r].append(error)
                    measures_std[key]['results'][r].append(np.std(error_local_list))
                    # print('key:', key, 'baseline:', baseline, 'error:', error, 'no access')
                # print('it:', it)
                it = it + 1

            else:
                base = per_dataset * which_dataset
                current_it = it
                for b_it in range(len(measures[key]['Baseline'])):
                    baseline = measures[key]['Baseline'][b_it]
                    b_indices = [baseline_indices[baseline][l] + base for l in range(len(baseline_indices[baseline]))]
                    error = 0
                    error_local_list = []
                    it = current_it
                    for ind in b_indices:
                        shapley_est = ShaplyzeEstimator(X=features_lists[ind_dataset],
                                                        causal_structure_info=causal_info if use_causal_info else [])
                        error += shapley_est.get_error(ID=it, folder_path=f'droup_out_res/dictionary_res_06_26',
                                                       baseline_dic_path=f'droup_out_res/dictionary_res_06_26/dic_{ind}.npy')
                        error_local_list.append(shapley_est.get_error(ID=it, folder_path=f'droup_out_res/dictionary_res_06_26',
                                                       baseline_dic_path=f'droup_out_res/dictionary_res_06_26/dic_{ind}.npy'))
                        it = it + 1
                    error = error / len(b_indices)
                    measures[key]['results'][r].append(error)
                    measures_std[key]['results'][r].append(np.std(error_local_list))
                    # print('key:', key, 'baseline:', baseline, 'error:', error, 'with access')
                    it = current_it + 5


    for key in measures.keys():
        for r in range(len(measures[key]['results'])):
            for b_it in range(len(measures[key]['Baseline'])):
                data_set_list.append(datasets[ind_dataset])
                measure_list.append(key)
                baseline_list.append(measures[key]['Baseline'][b_it])
                with_causal_info_list.append("YES")
                if measures[key]['surrogate'][r]:
                    with_access_list.append("YES")
                else:
                    with_access_list.append("NO")
                error_list.append(measures[key]['results'][r][b_it])
                error_std_list.append(measures_std[key]['results'][r][b_it])




import pandas as pd
dataframe = pd.DataFrame(
    {'Dataset': data_set_list, 'Measure': measure_list, 'Baseline': baseline_list, 'With Access': with_access_list, 'With Causal Info': with_causal_info_list, 'Error': error_list, 'Error STD': error_std_list})

# # 1. Keys with Baseline = "A"
baseline_A = [
    "$v_{A,CMI}$",
    "$v_{A,CMI_g}$",
    "$v_{A,DC}$",
    "$v_{A,HSIC}$",
    "$v_{A,NOCCO}$",
    "$v_{A,MMD}$",
    "$v_{A,Acc}$"
]


# 2. Keys with Baseline = "SP"
baseline_SP = [
    "$v_{SP,MI}$",
    "$v_{SP,MI_g}$",
    "$v_{SP,SI-A}$",
    "$v_{SP,SI_g-A}$",
    "$v_{SP,SI-X_S}$",
    "$v_{SP,SI_g-X_S}$",
    "$v_{D,MI.CMI.SI}$",
    "$v_{D,MI_g.CMI_g.SI_g}$",
    "$v_{D,MI.CMI}$",
    "$v_{D,MI_g.CMI_g}$",
    "$v_{SP,DC}$",
    "$v_{SP,HSIC}$",
    "$v_{SP,NOCCO}$",
    "$v_{SP,MMD}$",
    "$v_{SP,SI-A,\hat Y}$",
    "$v_{SP,SI_g-A,\hat Y}$",
    "$v_{SP,SPD}$"
]

# 3. Keys with Baseline = "EO"
baseline_EO = [
    "$v_{EO,CMI}$",
    "$v_{EO,CMI_g}$",
    "$v_{D,MI.CMI.SI}$",
    "$v_{D,MI_g.CMI_g.SI_g}$",
    "$v_{D,MI.CMI}$",
    "$v_{D,MI_g.CMI_g}$",
    "$v_{EO,CDC}$",
    "$v_{EO,CHSIC}$",
    "$v_{EO,CNOCCO}$",
    "$v_{EO,CMMD}$",
    "$v_{EO,EOD}$"
]

datasets = ["ACSIncome",
            "ACSEMployment",
            "ACSCoverage",
            "adult",
            "CensusIncomeKDD",
            "COMPAS",
            "Health"]


print('#############################################################')
print('Measure errors for different baseline with and without access to causal info')
print('#############################################################')

b_l = ['A', 'SP', 'EO']
chosen_b = [baseline_A, baseline_SP, baseline_EO]
seperators = ["Accuracy", "Statistical Parity", "Equalized Odds"]
for i in range(len(b_l)):
    print()
    print("-------------------------------------------------------------")
    print(seperators[i])
    print()
    print("measure & ", end="")
    for dataset in datasets:
        if dataset == datasets[-1]:
            print(f"{dataset} $X_S$ & {dataset} $\hat Y_{{|S}}$ \\\\")
        else:
            print(f"{dataset} $X_S$ & {dataset} $\hat Y_{{|S}}$  & ", end="")



    chosen_baseline = chosen_b[i]
    baseline= b_l[i]
    for measure in chosen_baseline:
        for dataset in datasets:
            if dataset == datasets[0]:
                print(f"{measure}", end=" & ")
            try:
                error_wiouth_access = dataframe[(dataframe["Dataset"] == dataset) & (dataframe["Measure"] == measure) & (dataframe["Baseline"] == baseline) & (dataframe["With Access"] == "NO") & (dataframe["With Causal Info"] == "NO")]["Error"].values[0]
                error_std_without_access = dataframe[(dataframe["Dataset"] == dataset) & (dataframe["Measure"] == measure) & (dataframe["Baseline"] == baseline) & (dataframe["With Access"] == "NO") & (dataframe["With Causal Info"] == "NO")]["Error STD"].values[0]
            except Exception as e:
                error_wiouth_access = "-"
                error_std_without_access = ""

            # print(dataframe[(dataframe["Dataset"] == dataset) & (dataframe["Measure"] == measure) & (dataframe["With Access"] == "NO")].head())
            # error_wiouth_access = dataframe[(dataframe["Dataset"] == dataset) & (dataframe["Measure"] == measure) & (dataframe["With Access"] == "NO") & (dataframe["With Causal Info"] == "NO")]["Error"].values[0]

            error_with_access = dataframe[(dataframe["Dataset"] == dataset) & (dataframe["Measure"] == measure) & (dataframe["Baseline"] == baseline) & (dataframe["With Access"] == "YES") & (dataframe["With Causal Info"] == "NO")]["Error"].values[0]
            error_std_with_access = dataframe[(dataframe["Dataset"] == dataset) & (dataframe["Measure"] == measure) & (dataframe["Baseline"] == baseline) & (dataframe["With Access"] == "YES") & (dataframe["With Causal Info"] == "NO")]["Error STD"].values[0]
            if error_wiouth_access =="-":
                print(f"- & {error_with_access*100:.1f} $\pm$ {error_std_with_access*100:.1f}", end="")
            elif error_wiouth_access< error_with_access:
                print(f"\\textbf{{{error_wiouth_access*100:.1f}  $\pm$ {error_std_without_access*100:.1f} }} & {error_with_access*100:.1f} $\pm$ {error_std_with_access*100:.1f}", end="")
            elif error_wiouth_access > error_with_access:
                print(f"{error_wiouth_access*100:.1f} $\pm$ {error_std_without_access*100:.1f} & \\textbf{{{error_with_access*100:.1f} $\pm$ {error_std_with_access*100:.1f}}}", end="")
            else:
                print(f"\\textbf{{{error_wiouth_access*100:.1f} $\pm$ {error_std_without_access*100:.1f}}} $\pm$ & \\textbf{{{error_with_access*100:.1f} $\pm$ {error_std_with_access*100:.1f}}} $\pm$", end="")
            if dataset == datasets[-1]:
                print(" \\\\")
            else:
                print(" & ", end="")
print()
print()
print('#############################################################')
print('Measure errors ratio with causal info / without causal info')
print('#############################################################')

b_l = ['A', 'SP', 'EO']

# # 1. Keys with Baseline = "A"
baseline_A = [
    "$v_{A,CMI}$",
    "$v_{A,CMI_g}$",
    "$v_{A,DC}$",
    "$v_{A,HSIC}$",
    "$v_{A,NOCCO}$",
    "$v_{A,MMD}$",
]


# 2. Keys with Baseline = "SP"
baseline_SP = [
    "$v_{SP,MI}$",
    "$v_{SP,MI_g}$",
    "$v_{SP,SI-A}$",
    "$v_{SP,SI_g-A}$",
    "$v_{SP,SI-X_S}$",
    "$v_{SP,SI_g-X_S}$",
    "$v_{D,MI.CMI.SI}$",
    "$v_{D,MI_g.CMI_g.SI_g}$",
    "$v_{D,MI.CMI}$",
    "$v_{D,MI_g.CMI_g}$",
    "$v_{SP,DC}$",
    "$v_{SP,HSIC}$",
    "$v_{SP,NOCCO}$",
    "$v_{SP,MMD}$",
    "$v_{SP,SI-A,\hat Y}$",
    "$v_{SP,SI_g-A,\hat Y}$",
]

# 3. Keys with Baseline = "EO"
baseline_EO = [
    "$v_{EO,CMI}$",
    "$v_{EO,CMI_g}$",
    "$v_{D,MI.CMI.SI}$",
    "$v_{D,MI_g.CMI_g.SI_g}$",
    "$v_{D,MI.CMI}$",
    "$v_{D,MI_g.CMI_g}$",
    "$v_{EO,CDC}$",
    "$v_{EO,CHSIC}$",
    "$v_{EO,CNOCCO}$",
    "$v_{EO,CMMD}$",
]





chosen_b = [baseline_A, baseline_SP, baseline_EO]
seperators = ["Accuracy", "Statistical Parity", "Equalized Odds"]
for i in range(len(b_l)):

    print()
    print("-------------------------------------------------------------")
    print(seperators[i])
    print()
    print("measure & ", end="")
    for dataset in datasets:
        if dataset == datasets[-1]:
            print(f"{dataset} $X_S$ & {dataset} $\hat Y_{{|S}}$ \\\\")
        else:
            print(f"{dataset} $X_S$ & {dataset} $\hat Y_{{|S}}$  & ", end="")

    chosen_baseline = chosen_b[i]
    baseline= b_l[i]
    for measure in chosen_baseline:
        for dataset in datasets:
            if dataset == datasets[0]:
                print(f"{measure}", end=" & ")
            try:
                error_wiouth_causal = dataframe[(dataframe["Dataset"] == dataset) & (dataframe["Measure"] == measure) & (dataframe["Baseline"] == baseline) & (dataframe["With Access"] == "NO") & (dataframe["With Causal Info"] == "NO")]["Error"].values[0]
            except Exception as e:
                error_wiouth_causal = dataframe[(dataframe["Dataset"] == dataset) & (dataframe["Measure"] == measure) & (dataframe["Baseline"] == baseline) & (dataframe["With Access"] == "YES") & (dataframe["With Causal Info"] == "NO")]["Error"].values[0]

            try:
                error_with_causal = dataframe[
                    (dataframe["Dataset"] == dataset) & (dataframe["Measure"] == measure) & (
                                dataframe["Baseline"] == baseline) & (dataframe["With Access"] == "NO") & (
                                dataframe["With Causal Info"] == "YES")]["Error"].values[0]
            except Exception as e:
                error_with_causal = dataframe[
                    (dataframe["Dataset"] == dataset) & (dataframe["Measure"] == measure) & (
                                dataframe["Baseline"] == baseline) & (dataframe["With Access"] == "YES") & (
                                dataframe["With Causal Info"] == "YES")]["Error"].values[0]

            ratio = error_with_causal / error_wiouth_causal
            # print(dataframe[(dataframe["Dataset"] == dataset) & (dataframe["Measure"] == measure) & (dataframe["With Access"] == "NO")].head())
            # error_wiouth_access = dataframe[(dataframe["Dataset"] == dataset) & (dataframe["Measure"] == measure) & (dataframe["With Access"] == "NO") & (dataframe["With Causal Info"] == "NO")]["Error"].values[0]
            if ratio<1:
                print(f"\\textbf{{{ratio*100:.0f}}}", end="")
            else:
                print(f"{ratio*100:.0f}", end="")
            if dataset == datasets[-1]:
                print(" \\\\")
            else:
                print(" & ", end="")



# Ensure Error is numeric
dataframe["Error"] = pd.to_numeric(dataframe["Error"], errors="coerce")
print()
print("#############################################################")
print('Percentage of improved measures with causal info for different datasets')
print("#############################################################")
print()
dataframe = dataframe[dataframe["With Access"] == "NO"]
for dataset, df_ds in dataframe.groupby("Dataset"):
    # Drop unwanted measures first
    df_filtered = df_ds[~df_ds["Measure"].isin(["Model accuracy", "SP difference", "EO difference"])]

    # Now split into YES and NO
    errors_yes = df_filtered[df_filtered["With Causal Info"] == "YES"]["Error"].values
    errors_no = df_filtered[df_filtered["With Causal Info"] == "NO"]["Error"].values


    # Compute ratio (YES/NO)
    ratio = errors_yes / errors_no
    count_decreased = np.sum(ratio < 1)
    print(dataset, ": percentage of improved measures= ", count_decreased/ len(ratio), '| #improved measures/#total number of measures =', count_decreased, "/",len(ratio))

print()
print('#############################################################')
print('Comparison between Gaussian and Original Mutual Information-based Measures (no causal info)')
print('#############################################################')
print()

# Define MI measures and their Gaussian counterparts
mi_pairs_A = [
    ("$v_{A,CMI}$", "$v_{A,CMI_g}$"),
]
mi_pairs_SP = [
    ("$v_{SP,MI}$", "$v_{SP,MI_g}$"),
    ("$v_{SP,SI-A}$", "$v_{SP,SI_g-A}$"),
    ("$v_{SP,SI-X_S}$", "$v_{SP,SI_g-X_S}$"),
    ("$v_{SP,SI-A,\\hat Y}$", "$v_{SP,SI_g-A,\\hat Y}$"),
]
mi_pairs_EO = [
    ("$v_{EO,CMI}$", "$v_{EO,CMI_g}$"),
    ("$v_{D,MI.CMI.SI}$", "$v_{D,MI_g.CMI_g.SI_g}$"),
    ("$v_{D,MI.CMI}$", "$v_{D,MI_g.CMI_g}$")
]
pair_dic = {'A': mi_pairs_A, 'SP': mi_pairs_SP, 'EO': mi_pairs_EO}

# --- NEW COUNTERS ---
improvement_counts = {dataset: 0 for dataset in datasets}
strong_improvement_counts = {dataset: 0 for dataset in datasets}  # ratio < 0.5
doubled_error_counts = {dataset: 0 for dataset in datasets}       # ratio >= 2
total_counts = {dataset: 0 for dataset in datasets}
# ---------------------

for baseline in ['A', 'SP', 'EO']:
    print()
    print(f"Baseline: {baseline}")
    print("-------------------------------------------------------------")
    print("Measure pair & ", end="")
    for dataset in datasets:
        if dataset == datasets[-1]:
            print(f"{dataset} ratio (%) \\\\")
        else:
            print(f"{dataset} ratio (%) & ", end="")
    mi_pairs = pair_dic[baseline]
    for orig, gauss in mi_pairs:
        print(f"{orig} vs {gauss}", end=" & ")
        for dataset in datasets:
            try:
                err_orig = dataframe[
                    (dataframe["Dataset"] == dataset) &
                    (dataframe["Measure"] == orig) &
                    (dataframe["Baseline"] == baseline) &
                    (dataframe["With Access"] == "NO") &
                    (dataframe["With Causal Info"] == "NO")
                ]["Error"].values[0]
                err_gauss = dataframe[
                    (dataframe["Dataset"] == dataset) &
                    (dataframe["Measure"] == gauss) &
                    (dataframe["Baseline"] == baseline) &
                    (dataframe["With Access"] == "NO") &
                    (dataframe["With Causal Info"] == "NO")
                ]["Error"].values[0]
                ratio = err_gauss / err_orig
                total_counts[dataset] += 1

                # --- Count categories ---
                if ratio < 1:
                    improvement_counts[dataset] += 1
                if ratio < 0.5:
                    strong_improvement_counts[dataset] += 1
                if ratio >= 2:
                    doubled_error_counts[dataset] += 1
                # -----------------------

                if ratio < 1:
                    print(f"\\textbf{{{ratio*100:.0f}}}", end="")
                else:
                    print(f"{ratio*100:.0f}", end="")
            except:
                print("-", end="")
            if dataset == datasets[-1]:
                print(" \\\\")
            else:
                print(" & ", end="")

# --- PRINT SUMMARY ---
print()
print("#############################################################")
print("Summary of Gaussian-based measure behavior per dataset")
print("#############################################################")
print()
for dataset in datasets:
    improved = improvement_counts[dataset]
    strong = strong_improvement_counts[dataset]
    doubled = doubled_error_counts[dataset]
    total = total_counts[dataset]
    perc_improved = (improved / total * 100) if total > 0 else 0
    perc_strong = (strong / total * 100) if total > 0 else 0
    perc_doubled = (doubled / total * 100) if total > 0 else 0
    print(f"{dataset}: improved={improved}/{total} ({perc_improved:.1f}%), "
          f"strongly improved={strong}/{total} ({perc_strong:.1f}%), "
          f"doubled error={doubled}/{total} ({perc_doubled:.1f}%)")


print()
print("#############################################################")
print("Average error variance (STD) per dataset and baseline (without access to prediction)")
print("#############################################################")
print()

# Filter out only the rows without access
df_no_access = dataframe[dataframe["With Access"] == "NO"]

# Compute mean Error STD per dataset and baseline
avg_std = (
    df_no_access.groupby(["Dataset", "Baseline"])["Error STD"]
    .mean()
    .reset_index()
    .pivot(index="Dataset", columns="Baseline", values="Error STD")
)

# Display results nicely
for dataset in avg_std.index:
    vals = avg_std.loc[dataset]
    print(
        f"{dataset}: "
        + ", ".join(
            f"{baseline}={vals[baseline]*100:.2f}%" if not pd.isna(vals[baseline]) else f"{baseline}= -"
            for baseline in ["A", "SP", "EO"]
        )
    )
# --- Compute average measure error (no access to predictions) for each baseline and dataset ---

avg_errors = {}

for dataset in datasets:
    avg_errors[dataset] = {}
    for baseline in ['A', 'SP', 'EO']:
        # Select relevant rows
        subset = dataframe[
            (dataframe["Dataset"] == dataset) &
            (dataframe["Baseline"] == baseline) &
            (dataframe["With Access"] == "NO")
        ]

        # Compute mean of "Error"
        if not subset.empty:
            avg_error = subset["Error"].mean()
            avg_errors[dataset][baseline] = avg_error
        else:
            avg_errors[dataset][baseline] = np.nan

# Print results nicely
print("\nAverage measure errors (no access to predictions):")
for dataset, vals in avg_errors.items():
    print(f"{dataset}: " +
          f"A={vals.get('A', np.nan)*100:.2f}%, " +
          f"SP={vals.get('SP', np.nan)*100:.2f}%, " +
          f"EO={vals.get('EO', np.nan)*100:.2f}%")
