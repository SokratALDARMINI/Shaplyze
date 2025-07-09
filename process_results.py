import numpy as np
from folktables import ACSIncome

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
#     "Accuracy": {'Baseline': ['A_ref'], 'bins': False, 'surrogate':[False], 'seeds': [seeds_list], 'results':[[]]},
#     "SP": {'Baseline': ['SP_ref'], 'bins': False, 'surrogate':[False], 'seeds': [seeds_list], 'results':[[]]},
#     "EO": {'Baseline': ['EO_ref'], 'bins': False, 'surrogate':[False], 'seeds': [seeds_list], 'results':[[]]},
# }
with_access = []
without_access = []
target = 'EO'
datasets = ["ACSIncome",
            "ACSCoverage",
            "ACSEMployment",
            "COMPAS",
            "CensusIncomeKDD",
            "adult",
            "Health"]

order_over_datasets = [0, 2, 1, 5, 4, 3, 6]  # Order of datasets in the results
# order_over_datasets = [0]  # Order of datasets in the results
for ind_dataset in order_over_datasets:
    print(f"Dataset: {datasets[ind_dataset]}")

    measures = {
        "I(A;X_S)": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                     'results': [[], []]},
        "I(Y;X_S|A,X_{S^C})": {'Baseline': ['A'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                               'results': [[], []]},
        "SI(A;X_S,Y)": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                        'results': [[], []]},
        "SI(X_S;A,Y)": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                        'results': [[], []]},
        "I(A;X_S|Y)": {'Baseline': ['EO'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                       'results': [[], []]},
        "I(A;X_S)I(A;X_S|Y)SI(Y;X_S,A)": {'Baseline': ['SP', 'EO'], 'bins': True, 'surrogate': [False, True],
                                          'seeds': [None, seeds_list], 'results': [[], []]},
        "I(A;X_S)I(A;X_S|Y)": {'Baseline': ['SP', 'EO'], 'bins': True, 'surrogate': [False, True],
                               'seeds': [None, seeds_list], 'results': [[], []]},
        "V^2(X_S,A)": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                       'results': [[], []]},
        "V^2(X_S,Y)": {'Baseline': ['A'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                       'results': [[], []]},
        "V^2_Y(X_S,A)": {'Baseline': ['EO'], 'bins': True, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                         'results': [[], []]},
        "I_g(A;X_S)": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                       'results': [[], []]},
        "I_g(Y;X_S|A,X_{S^C})": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                                 'seeds': [None, seeds_list], 'results': [[], []]},
        "SI_g(A;X_S,Y)": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                          'results': [[], []]},
        "SI_g(X_S;A,Y)": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                          'results': [[], []]},
        "I_g(A;X_S|Y)": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                         'results': [[], []]},
        "I_g(A;X_S)I_g(A;X_S|Y)SI_g(Y;X_S,A)": {'Baseline': ['SP', 'EO'], 'bins': False, 'surrogate': [False, True],
                                                'seeds': [None, seeds_list], 'results': [[], []]},
        "I_g(A;X_S)I_g(A;X_S|Y)": {'Baseline': ['SP', 'EO'], 'bins': False, 'surrogate': [False, True],
                                   'seeds': [None, seeds_list], 'results': [[], []]},
        "HSIC(X_S,Y)": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                        'results': [[], []]},
        "NOCCO(X_S,Y)": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                         'results': [[], []]},
        "HSIC(X_S,A)": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                        'results': [[], []]},
        "NOCCO(X_S,A)": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                         'results': [[], []]},
        "HSIC_Y(X_S,A)": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                          'results': [[], []]},
        "NOCCO_Y(X_S,A)": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True], 'seeds': [None, seeds_list],
                           'results': [[], []]},
        "MMD^2(X_{S,Y=0},X_{S,Y=1})": {'Baseline': ['A'], 'bins': False, 'surrogate': [False, True],
                                       'seeds': [None, seeds_list], 'results': [[], []]},
        "MMD^2(X_{S,A=0},X_{S,A=1})": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                                       'seeds': [None, seeds_list], 'results': [[], []]},
        "MMD^2_Y(X_{S,A=0},X_{S,A=1})": {'Baseline': ['EO'], 'bins': False, 'surrogate': [False, True],
                                         'seeds': [None, seeds_list], 'results': [[], []]},
        "SI(A;X_S,hat Y)": {'Baseline': ['SP'], 'bins': True, 'surrogate': [False, True],
                            'seeds': [seeds_list, seeds_list], 'results': [[], []]},
        "SI_g(A;X_S,hat Y)": {'Baseline': ['SP'], 'bins': False, 'surrogate': [False, True],
                              'seeds': [seeds_list, seeds_list], 'results': [[], []]},
        "Accuracy": {'Baseline': ['A_ref'], 'bins': False, 'surrogate': [False], 'seeds': [seeds_list],
                     'results': [[]]},
        "SP": {'Baseline': ['SP_ref'], 'bins': False, 'surrogate': [False], 'seeds': [seeds_list], 'results': [[]]},
        "EO": {'Baseline': ['EO_ref'], 'bins': False, 'surrogate': [False], 'seeds': [seeds_list], 'results': [[]]},
    }

    which_dataset = ind_dataset

    per_dataset = 191


    total = per_dataset
    it = per_dataset * which_dataset
    previous_sh = 0
    for key in measures.keys():
        for r in range(len(measures[key]['results'])):
            if measures[key]['seeds'][r]  is None:
                try:
                    sh = np.load(f'results/Shapley_values_{it}.npy', allow_pickle=True)
                except FileNotFoundError:
                    sh = [1, 1, 1, 1]
                    print(f"File results/Shapley_values_{it}.npy not found.")

                measures[key]['results'][r].append(sh)
                measures[key]['results'][r].append(sh)
                measures[key]['results'][r].append(sh)
                measures[key]['results'][r].append(sh)
                measures[key]['results'][r].append(sh)
                it = it + 1
            else:
                for s in range(5):
                    try:
                        sh = np.load(f'results/Shapley_values_{it}.npy', allow_pickle=True)
                    except FileNotFoundError:
                        sh = [1, 1, 1, 1]
                        print(f"File results/Shapley_values_{it}.npy not found.")
                    measures[key]['results'][r].append(sh)
                    it = it + 1

    for key in measures.keys():
        for r in range(len(measures[key]['results'])):
            for i in range(5):
                measures[key]['results'][r][i] = np.array(measures[key]['results'][r][i])
                # standardize the results by dividing by the maximum absolute value
                measures[key]['results'][r][i] = measures[key]['results'][r][i] / np.max(np.abs(measures[key]['results'][r][i]))



    baseline_pairs = {'A': 'Accuracy', 'SP': 'SP', 'EO': 'EO'}
    target_pair = baseline_pairs[target]

    replacement= [False, True]  # Replacement with X_S or not
    for m in range(2):
        print("--------------------------")
        rep = replacement[m]
        if rep:
            print(f'with access to hat Y S')
        else:
            print(f'Using X_S')
        print()
        for key in measures.keys():
            if target in measures[key]['Baseline']:
                error = 0
                for r in range(len(measures[key]['results'])):
                    for i in range(5):
                        error = error + np.mean(np.abs(measures[key]['results'][r][i] - measures[target_pair]['results'][0][i]))
                    error = error / 5.0
                    if measures[key]['surrogate'][r]: # replacement with X_S
                        # round error for 3 decimal places
                        if rep:
                            # print(f'{key}(surrogate): {error:.3f}')
                            pass

                            print(f'{error:.3f}')
                            with_access.append(error)


                    else: # without replacement with X_S
                        if not rep:
                            # print(f'{key}')



                            print(f'{error:.3f}')
                            without_access.append(error)




# devide the results into two lists: with access to hat Y and without access to hat Y
with_access = np.array(with_access)
without_access = np.array(without_access)
ratio = with_access / without_access
# compute percentage of improvement
percentage_improvement = (without_access - with_access) / without_access * 100

# compute the how many improved by 10 %
improved_by_10 = np.sum(percentage_improvement > 10)/ len(percentage_improvement) * 100
print(f"Percentage of improvement by more than 10%: {improved_by_10:.2f}%")
# compute the how many improved by 20 %
improved_by_20 = np.sum(percentage_improvement > 20)/ len(percentage_improvement) * 100
print(f"Percentage of improvement by more than 20%: {improved_by_20:.2f}%")
# compute the how many improved by 30 %
improved_by_30 = np.sum(percentage_improvement > 30)/ len(percentage_improvement) * 100
print(f"Percentage of improvement by more than 30%: {improved_by_30:.2f}%")
# compute the how many improved by 40 %
improved_by_40 = np.sum(percentage_improvement > 40)/ len(percentage_improvement) * 100
print(f"Percentage of improvement by more than 40%: {improved_by_40:.2f}%")
# compute the how many improved by 50 %
improved_by_50 = np.sum(percentage_improvement > 50)/ len(percentage_improvement) * 100
print(f"Percentage of improvement by more than 50%: {improved_by_50:.2f}%")
# compute the how many improved by 60 %
improved_by_60 = np.sum(percentage_improvement > 60)/ len(percentage_improvement) * 100
print(f"Percentage of improvement by more than 60%: {improved_by_60:.2f}%")
# compute the how many improved by 70 %
improved_by_70 = np.sum(percentage_improvement > 70)/ len(percentage_improvement) * 100
print(f"Percentage of improvement by more than 70%: {improved_by_70:.2f}%")
# compute the how many improved by 80 %
improved_by_80 = np.sum(percentage_improvement > 80)/ len(percentage_improvement) * 100
print(f"Percentage of improvement by more than 80%: {improved_by_80:.2f}%")