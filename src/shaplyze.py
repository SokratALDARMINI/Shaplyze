import itertools
import numpy as np
import math

class ShaplyzeEstimator:
    def __init__(self, measure, causal_structure_info = []):
        self.measure = measure
        self.X = measure.X
        self.Y = measure.Y
        self.A = measure.A
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
        self.Beta = list(zip(all_permutations, weights))

    def phi_i_asy(self, Xi):
        for xi in Xi:  # Check if the features are in the list of features
            assert xi in self.X, "Feature {} not in features".format(xi)
        phi = 0

        for perm, weight in self.Beta:
            index = perm.index(Xi[0])
            subset_with_xi = perm[:index + 1]
            subset_without_xi = perm[:index]

            if weight == 0:
                continue
            phi = phi + (self.dic[frozenset(subset_with_xi)] - self.dic[frozenset(subset_without_xi)]) * weight

        return phi  # Return the accuracy and discrimination impacts

    def get_sh_values_save(self, path=None, subset_id=0):

        subsets = self.subsets(self.X)
        id = 0
        for subset in subsets:
            if id == subset_id:
                subset_set = frozenset(subset)
                self.dic[subset_set] = self.measure.evaluate(subset)
                # save the dictionary to the specified path as .npy file
                # create the folder if it does not exist, then save the dictionary with the id as name
                if path is not None:
                    import os
                    if not os.path.exists(path):
                        os.makedirs(path)
                    np.save(os.path.join(path, f'subset_{id}.npy'), self.dic)


                return 0
            else:
                id += 1
                continue


    def get_sh_values(self):

        subsets = self.subsets(self.X)
        for subset in subsets:
            subset_set = frozenset(subset)
            if subset_set not in self.dic:
                self.dic[subset_set] = self.measure.evaluate(subset)

        sh_values = []
        for xi in self.X:
            sh_values.append(self.phi_i([xi]))

        return sh_values  # Return the Shapley values for each feature

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
    def comp(self, Xs):  # Compute the complementary set of Xs
        return [x for x in self.X if x not in Xs]

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