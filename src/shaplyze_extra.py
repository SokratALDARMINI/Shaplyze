import itertools
import numpy as np
import math
import os

class ShaplyzeEstimator:
    def __init__(self, X = [], causal_structure_info = []):
        self.X = X
        self.n = len(self.X)  # Number of features
        self.dic = {}
        for i in range(len(causal_structure_info)):
            assert isinstance(causal_structure_info[i], list), f"Causal info {causal_structure_info[i]} is not a list"
            assert len(causal_structure_info[i]) == 2, f"Causal info {causal_structure_info[i]} is not a list of two features"
            for j in range(2):
                assert causal_structure_info[i][j] in self.X, f"Causal info {causal_structure_info[i][j]} not in features list"

        all_permutations = [list(p) for p in itertools.permutations(self.X)]
        weights = [1.0 for _ in range(len(all_permutations))]
        for k in range(len(all_permutations)):
            perm = all_permutations[k]
            for pair in causal_structure_info:
                if perm.index(pair[0]) >= perm.index(pair[1]):
                    weights[k] = 0.0
                    break
        weights = np.array(weights)
        weights /= np.sum(weights)
        # if len(causal_structure_info)>0:
            # print('max weight', np.max(weights))
            # print('min weight', np.min(weights))
        self.Beta = list(zip(all_permutations, weights))

    def phi_i_asy(self, Xi):
        for xi in Xi:  # Check if the features are in the list of features
            assert xi in self.X, "Feature {} not in features".format(xi)
        phi = 0

        for perm, weight in self.Beta:
            index = perm.index(Xi[0])
            subset_with_xi = perm[:index + 1]
            subset_without_xi = perm[:index]

            if weight == 0.0:
                continue
            phi = phi + (self.dic[frozenset(subset_with_xi)] - self.dic[frozenset(subset_without_xi)]) * weight

        return phi  # Return the accuracy and discrimination impacts

    def get_sh_values(self, read_dic = False, ID=0,folder_path='dictionary_res'):
        if read_dic:
            # Load dictionary from .npy file
            load_path = os.path.join(folder_path, f"dic_{ID}.npy")
            self.dic = np.load(load_path, allow_pickle=True).item()
            # print(f"Dictionary loaded from {load_path}")
            sh_values = []
            for xi in self.X:
                sh_values.append(self.phi_i_asy([xi]))

            return sh_values


    def comp(self, Xs):  # Compute the complementary set of Xs
        return [x for x in self.X if x not in Xs]


    def calcSubset(self, features, res, subset,
                   index):  # Generate all possible subsets of a set of features (This is a recursive function)
        # Add the current subset to the result list
        res.append(subset[:])

        # Generate subsets by recursively including and excluding elements
        for i in range(index, len(features)):
            # Include the current element in the subset
            subset.append(features[i])

            # Recursively generate subsets with the current element included
            self.calcSubset(features, res, subset, i + 1)

            # Exclude the current element from the subset (backtracking)
            subset.pop()

    def subsets(self, features):  # Generate all possible subsets of a set of features
        subset = []
        res = []
        index = 0
        self.calcSubset(features, res, subset, index)
        return res

    def phi_i(self, Xi):
        for xi in Xi:  # Check if the features are in the list of features
            assert xi in self.X, "Feature {} not in features".format(xi)
        phi_i = 0
        comp_Xi = self.comp(Xi)  # Compute the complementary set of Xi
        subsets_comp_Xi = self.subsets(comp_Xi)  # Generate all possible subsets of the complementary set of Xi
        for subset in subsets_comp_Xi:  # Compute the accuracy and discrimination impacts of the subsets of the complementary set of Xi
            c = (math.factorial(len(subset)) * math.factorial(self.n - len(subset) - 1) / (
                math.factorial(self.n)))  # Compute the coefficient
            phi_i += (self.dic[frozenset(subset + Xi)] - self.dic[
                frozenset(subset)]) * c  # aggregate the accuracy impacts
            # if ((self.dic_vd[frozenset(subset + Xi)] - self.dic_vd[frozenset(subset)])<0):
            #     print('subset =', subset, 'Xi =', Xi, 'subset + Xi =', subset + Xi, 'vd+1 =', self.dic_vd[frozenset(subset + Xi)], 'vd =', self.dic_vd[frozenset(subset)])

        return phi_i  # Return the accuracy and discrimination impacts

    def get_baseline(self, baseline_dic_path):
        self.dic = np.load(baseline_dic_path, allow_pickle=True).item()
        # print('baseline_dic_path', baseline_dic_path)
        # print('dic', self.dic)
        baseline = []
        baseline_ref = self.dic[frozenset(self.X)]
        for xi in self.X:
            baseline.append(baseline_ref-self.dic[frozenset(self.comp([xi]))])
        return baseline

    def get_error(self,ID=0,folder_path='dictionary_res', baseline_dic_path=None):
        SVs = self.get_sh_values(read_dic=True, ID=ID, folder_path=folder_path)
        # print('SVs',SVs)
        baseline = self.get_baseline(baseline_dic_path)
        # print('baseline',baseline)
        SVs = np.array(SVs)
        baseline = np.array(baseline)
        SVs = SVs/np.max(np.abs(SVs))
        baseline = baseline/np.max(np.abs(baseline))
        error = np.abs(SVs-baseline)
        return np.mean(error)/2
