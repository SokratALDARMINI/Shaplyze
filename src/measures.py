from abc import ABC, abstractmethod
import numpy as np
from dit.npdist import Distribution
import pandas as pd
from typing import Literal
from sklearn.linear_model import LogisticRegression

from src.midist import MIDist as I_dis
from src.migaussian import MIGaussian as I_gu
from src.hsic import HSIC
from src.mmd import MMD
from src.distancecov import DistanceCovariance as DistCov
from src.classifier import ClassifierBase as ClassifierBase

# Base class for all measures
class Measure(ABC):
    @abstractmethod
    def evaluate(self, subset: list[str]) -> float:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

# Measure for information theoretic measures based on distributions
class InfoDistributionMeasure(Measure):
    def __init__(self, df, X, A, Y, bins= (), eps = 1e-7):
        self.X = X
        self.A = A
        self.Y = Y
        self.bins = bins
        self.eps = eps
        self.header = X + A + Y
        data = df[self.X + self.A + self.Y].values
        hist, edges = np.histogramdd(data, bins=self.bins)
        self.dist = Distribution.from_ndarray(hist / df.shape[0])

class I_Xs_A(InfoDistributionMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        return I_dis.mi(self.dist, self.header, Xs, self.A)

    def name(self):
        return "I(Xs;A)"

class I_Y_Xs_given_AXsc(InfoDistributionMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0 or len(self.X) == 0:
            return 0.0
        Xsc = [x for x in self.X if x not in Xs]
        cond = self.A + Xsc
        return I_dis.cmi(self.dist, self.header, self.Y, Xs, cond)

    def name(self):
        return "I(Y;Xs|A,Xsc)"

class SI_A_Xs_Y(InfoDistributionMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        # print("SI_A_Xs_Y evaluate called with Xs:", Xs)
        if len(Xs) == 0:
            return 0.0
        return I_dis.red(self.dist, self.header, self.A, Xs, self.Y)

    def name(self):
        return "SI(A;Xs,Y)"

class SI_Xs_A_Y(InfoDistributionMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        return I_dis.red(self.dist, self.header, Xs, self.A, self.Y)

    def name(self):
        return "SI(Xs;A,Y)"

class I_Xs_A_given_Y(InfoDistributionMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        return I_dis.cmi(self.dist, self.header, Xs, self.A, self.Y)

    def name(self):
        return "I(Xs;A|Y)"

class I_A_Xs_times_IAXs_given_Y_times_SI_Y_Xs_A(InfoDistributionMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        i1 = I_dis.mi(self.dist, self.header, Xs, self.A)
        i2 = I_dis.cmi(self.dist, self.header, Xs, self.A, self.Y)
        ri = I_dis.red(self.dist, self.header, self.Y, Xs, self.A)
        return i1 * i2 * ri

    def name(self):
        return "I(A;Xs)×I(A;Xs|Y)×SI(Y;Xs,A)"

class I_A_Xs_times_IAXs_given_Y(InfoDistributionMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        i1 = I_dis.mi(self.dist, self.header, Xs, self.A)
        i2 = I_dis.cmi(self.dist, self.header, Xs, self.A, self.Y)
        return i1 * i2

    def name(self):
        return "I(A;Xs)×I(A;Xs|Y)"

# Measure for information theoretic measures based on Distance Covariance
class DistCovMeasure(Measure):
    def __init__(self, df, X, A, Y, bins= (), eps = 1e-7):
        self.X = X
        self.A = A
        self.Y = Y
        self.bins = bins
        self.header = X + A + Y
        data = df[self.X + self.A + self.Y].values
        hist, edges = np.histogramdd(data, bins=self.bins)
        self.dist = Distribution.from_ndarray(hist / df.shape[0])

class DisCov_Xs_A(InfoDistributionMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        return DistCov.distanceCov(self.dist, self.header, Xs, self.A)

    def name(self):
        return "DisCov(Xs;A)"

class DisCov_Xs_Y(InfoDistributionMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        return DistCov.distanceCov(self.dist, self.header, Xs, self.Y)

    def name(self):
        return "DisCov(Xs;Y)"

class DisCov_Xs_A_given_Y(InfoDistributionMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        return DistCov.conditionalDistanceCov(self.dist, self.header, Xs, self.A, self.Y)
    def name(self):
        return "DisCov_Y(Xs;A)"

# Measure for information theoretic measures based on Gaussian assumption
class InfoGaussianMeasure(Measure):
    def __init__(self, df, X, A, Y, eps=1e-6):
        self.df = df
        self.X = X
        self.A = A
        self.Y = Y
        self.header = X + A + Y
        self.eps = eps

class Ig_Xs_A(InfoGaussianMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        return I_gu.mi(self.df, Xs, self.A, eps=self.eps)

    def name(self):
        return "Ig(Xs;A)"

class Ig_Y_Xs_given_AXsc(InfoGaussianMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0 or len(self.X) == 0:
            return 0.0
        Xsc = [x for x in self.X if x not in Xs]
        cond = self.A + Xsc
        return I_gu.cmi(self.df, self.Y, Xs, cond, eps=self.eps)

    def name(self):
        return "Ig(Y;Xs|A,Xsc)"

class SIg_A_Xs_Y(InfoGaussianMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        return I_gu.red(self.df, self.A, Xs, self.Y, eps=self.eps)

    def name(self):
        return "SIg(A;Xs,Y)"

class SIg_Xs_A_Y(InfoGaussianMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        return I_gu.red(self.df, Xs, self.A, self.Y, eps=self.eps)

    def name(self):
        return "SIg(Xs;A,Y)"

class Ig_Xs_A_given_Y(InfoGaussianMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        return I_gu.cmi(self.df, Xs, self.A, self.Y, eps=self.eps)

    def name(self):
        return "Ig(Xs;A|Y)"

class Ig_A_Xs_times_IAXs_given_Y_times_SI_Y_Xs_A(InfoGaussianMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        i1 = I_gu.mi(self.df, Xs, self.A, eps=self.eps)
        i2 = I_gu.cmi(self.df, Xs, self.A, self.Y, eps=self.eps)
        ri = I_gu.red(self.df, self.Y, Xs, self.A, eps=self.eps)
        return i1 * i2 * ri

    def name(self):
        return "Ig(A;Xs)×Ig(A;Xs|Y)×SIg(Y;Xs,A)"

class Ig_A_Xs_times_IAXs_given_Y(InfoGaussianMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        i1 = I_gu.mi(self.df, Xs, self.A, eps=self.eps)
        i2 = I_gu.cmi(self.df, Xs, self.A, self.Y, eps=self.eps)
        return i1 * i2

    def name(self):
        return "Ig(A;Xs)×Ig(A;Xs|Y)"

# Measure for kernel-based dependence measures
class HSICMeasure(Measure):
    def __init__(self, df, X, A, Y, gamma_X=0.5, gamma_Y=0.5, eps=1e-6):
        self.df = df
        self.A = A
        self.Y = Y
        self.X = X
        self.gamma_X = gamma_X
        self.gamma_Y = gamma_Y
        self.eps = eps

class HSIC_Xs_Y(HSICMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0

        return HSIC.hsic(self.df ,X= Xs, Y= self.Y, gamma_X=self.gamma_X, gamma_Y=self.gamma_Y)

    def name(self):
        return "HSIC(Xs;Y)"

class NOCCO_Xs_Y(HSICMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0

        return HSIC.nocco(self.df ,X= Xs, Y= self.Y,  gamma_X=self.gamma_X, gamma_Y=self.gamma_Y, eps=self.eps)

    def name(self):
        return "NOCCO(Xs;Y)"

class HSIC_Xs_A(HSICMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0

        return HSIC.hsic(self.df, X= Xs, Y= self.A, gamma_X=self.gamma_X, gamma_Y=self.gamma_Y)

    def name(self):
        return "HSIC(Xs;A)"

class NOCCO_Xs_A(HSICMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        X = self.df[Xs].values
        A = self.df[self.A].values
        return HSIC.nocco(self.df, X= Xs, Y= self.A, gamma_X=self.gamma_X, gamma_Y=self.gamma_Y, eps=self.eps)

    def name(self):
        return "NOCCO(Xs;A)"

class HSIC_Xs_A_given_Y(HSICMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0

        return HSIC.chsic(self.df, X_cols= Xs, Y_cols= self.A, Z_cols = self.Y, gamma_X=self.gamma_X, gamma_Y=self.gamma_Y)

    def name(self):
        return "HSIC_Y(Xs;A)"

class NOCCO_Xs_A_given_Y(HSICMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0

        return HSIC.cnocco(self.df, X_cols= Xs, Y_cols= self.A, Z_cols = self.Y, gamma_X=self.gamma_X, gamma_Y=self.gamma_Y)

    def name(self):
        return "NOCCO_Y(Xs;A)"

# Measure for based on Maximum Mean Discrepancy (MMD)
class MMDMeasure(Measure):
    def __init__(self, df, X, A, Y, gamma=0.5):
        self.df = df
        self.A = A
        self.X = X
        self.Y = Y
        self.gamma = gamma

class MMD_Xs_Y(MMDMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        return MMD.mmd(self.df, Y=self.Y, X=Xs, gamma=self.gamma)

    def name(self):
        return "MMD(Xs;Y)"

class MMD_Xs_A(MMDMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        return MMD.mmd(self.df, Y=self.A, X=Xs, gamma=self.gamma)

    def name(self):
        return "MMD(Xs;A)"

class MMD_Xs_A_given_Y(MMDMeasure):
    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        return MMD.cmmd(self.df, Z=self.A, A=Xs, B=self.Y, gamma=self.gamma)

    def name(self):
        return "MMD_Y(Xs;A)"

# Measure for classifier-based measures
class ClassifierMeasure(Measure, ABC):
    def __init__(self, df, X, A, Y, mode: Literal["replace", "drop"],classifier = LogisticRegression(), categorical_features = [], split_ratio = 0.33, seed = 0, **kwargs):
        if 'fit_params' in kwargs:
            fit_params = kwargs['fit_params']
            kwargs.pop('fit_params')
        else:
            fit_params = {}
        self.classifier_base = ClassifierBase(classifier, df, X, A, Y, categorical_features=categorical_features, split_ratio=split_ratio, seed=seed, **fit_params)
        self.mode = mode
        self.X = X
        self.A = A
        self.Y = Y

    def _get_df(self, subset: list[str]) -> pd.DataFrame:
        if self.mode == "replace":
            return self.classifier_base.classify_replace_mean(subset)
        else:
            return self.classifier_base.classify_drop_out(subset)

class AccuracyMeasure(ClassifierMeasure):
    def __init__(self, df, X, A, Y,
                 mode: Literal["replace", "drop"],
                 classifier=LogisticRegression(),
                 categorical_features=[],
                 split_ratio=0.33,
                 seed=0, **kwargs):
        super().__init__(df, X, A, Y,
                         mode=mode, classifier=classifier,
                         categorical_features=categorical_features,
                         split_ratio=split_ratio, seed=seed, **kwargs)

        df_new = self._get_df([])

        self.acc_ref = (df_new['Yp'] == df_new[self.classifier_base.target_label[0]]).mean()


    def evaluate(self, subset: list[str]) -> float:
        df_new = self._get_df(subset)
        # print('subset', subset)
        # print('acc', (df_new['Yp'] == df_new[self.classifier_base.target_label[0]]).mean())
        # print(df_new['Yp'])
        # print(df_new[self.classifier_base.target_label[0]])
        # exit()
        # return (df_new['Yp'] == df_new[self.classifier_base.target_label[0]]).mean()
        return (df_new['Yp'] == df_new[self.classifier_base.target_label[0]]).mean() - self.acc_ref

    def name(self) -> str:
        return f"Accuracy({self.mode})"

class DPMeasure(ClassifierMeasure):
    def evaluate(self, subset: list[str]) -> float:
        if len(subset) == 0:
            return 0.0
        df_new = self._get_df(subset)
        A_col = self.classifier_base.sensitive_attribute[0]
        group_0 = df_new[df_new[A_col] == 0]['Yp'].mean()
        group_1 = df_new[df_new[A_col] == 1]['Yp'].mean()
        return abs(group_0 - group_1)

    def name(self) -> str:
        return f"DP({self.mode})"

class EOMeasure(ClassifierMeasure):
    def evaluate(self, subset: list[str]) -> float:
        if len(subset) == 0:
            return 0.0
        df_new = self._get_df(subset)
        A_col = self.classifier_base.sensitive_attribute[0]
        Y_col = self.classifier_base.target_label[0]
        #  For Y=1
        group_0_1 = df_new[(df_new[A_col] == 0) & (df_new[Y_col] == 1)]['Yp'].mean()
        group_1_1 = df_new[(df_new[A_col] == 1) & (df_new[Y_col] == 1)]['Yp'].mean()

        #  For Y=0
        group_0_0 = df_new[(df_new[A_col] == 0) & (df_new[Y_col] == 0)]['Yp'].mean()
        group_1_0 = df_new[(df_new[A_col] == 1) & (df_new[Y_col] == 0)]['Yp'].mean()
        # Calculate the difference in outcomes
        # compute positive rate

        PR = df_new[df_new[Y_col] == 1].shape[0] / df_new.shape[0]
        # EO difference between A=0 and A=1 groups

        return abs(group_0_1 - group_1_1)* PR + abs(group_0_0 - group_1_0)*(1-PR)

    def name(self) -> str:
        return f"EO({self.mode})"

class EOPY1Measure(ClassifierMeasure):
    def evaluate(self, subset: list[str]) -> float:
        if len(subset) == 0:
            return 0.0
        df_new = self._get_df(subset)
        A_col = self.classifier_base.sensitive_attribute[0]
        Y_col = self.classifier_base.target_label[0]

        # Only consider samples where Y=1 (advantaged outcome)
        df_new = df_new[df_new[Y_col] == 1]

        # EO difference between A=0 and A=1 groups
        group_0 = df_new[df_new[A_col] == 0]['Yp'].mean()
        group_1 = df_new[df_new[A_col] == 1]['Yp'].mean()

        return abs(group_0 - group_1)

    def name(self) -> str:
        return f"EOPY1({self.mode})"

class EOPY0Measure(ClassifierMeasure):
    def evaluate(self, subset: list[str]) -> float:
        if len(subset) == 0:
            return 0.0
        df_new = self._get_df(subset)
        A_col = self.classifier_base.sensitive_attribute[0]
        Y_col = self.classifier_base.target_label[0]

        # Only consider samples where Y=0 (advantaged outcome)
        df_new = df_new[df_new[Y_col] == 0]

        # EO difference between A=0 and A=1 groups
        group_0 = df_new[df_new[A_col] == 0]['Yp'].mean()
        group_1 = df_new[df_new[A_col] == 1]['Yp'].mean()

        return abs(group_0 - group_1)

    def name(self) -> str:
        return f"EOPY0({self.mode})"

class PPMeasure(ClassifierMeasure):
    def evaluate(self, subset: list[str]) -> float:
        if len(subset) == 0:
            return 0.0
        df_new = self._get_df(subset)
        A_col = self.classifier_base.sensitive_attribute[0]
        Y_col = self.classifier_base.target_label[0]

        # Filter for predicted positive samples
        df_new = df_new[df_new['Yp'] == 1]

        # If no predicted positives, return 0 to avoid NaNs
        if df_new.empty:
            return 0.0

        # Compute P(Y=1 | Yp=1, A=a) for a=0 and a=1
        group_0 = df_new[df_new[A_col] == 0][Y_col].mean()
        group_1 = df_new[df_new[A_col] == 1][Y_col].mean()

        return abs(group_0 - group_1)

    def name(self) -> str:
        return f"PP({self.mode})"  # PP = Predictive Parity

class SurrogateMeasure(Measure):
    def __init__(self, df, X, A, Y, measure_class: type, mode:  Literal["replace", "drop"], categorical_features = [], classifier = LogisticRegression(), split_ratio = 0.33, seed = 0, **kwargs):
        if 'fit_params' in kwargs:
            fit_params = kwargs['fit_params']
            kwargs.pop('fit_params')
        else:
            fit_params = {}
        self.classifier_base =  ClassifierBase(classifier, df, features=X, sensitive_attribute=A, target_label=Y, categorical_features=categorical_features, split_ratio=split_ratio, seed=seed, **fit_params)
        self.measure_class = measure_class
        self.mode = mode
        self.measure_kwargs = kwargs
        self.X = X
        self.A = A
        self.Y = Y
        self.header = X + A + Y
        self.measure_kwargs['X'] = X
        self.measure_kwargs['A'] = A
        self.measure_kwargs['Y'] = Y


    def evaluate(self, subset: list[str]) -> float:
        if len(subset) == 0:
            return 0.0
        df_new = self.classifier_base.classify_replace_mean(subset) if self.mode == "replace" else self.classifier_base.classify_drop_out(subset)
        measure_kwargs = self.measure_kwargs.copy()
        indices_for_removal = []
        for index in range(len(self.measure_kwargs['X'])):
            if self.measure_kwargs['X'][index] in subset:
                indices_for_removal.append(index)
                # drop the column from the features to avoid redundancy
                df_new = df_new.drop(columns=[self.measure_kwargs['X'][index]])

        if 'bins' in measure_kwargs:
            # remove indices for removal from the bins
            measure_kwargs['bins'] = tuple([b for i, b in enumerate(measure_kwargs['bins']) if i not in indices_for_removal])
        # remove subset from the features
        measure_kwargs['X'] = [x for x in self.measure_kwargs['X'] if x not in subset]
        # add 'Yp' to the features for evaluation
        measure_kwargs['X'] = measure_kwargs['X'] + ['Yp']  # Ensure 'Yp' is included in the features for evaluation
        # Add bin for 'Yp'
        if 'bins' in self.measure_kwargs:
            measure_kwargs['bins'] = tuple(list(measure_kwargs['bins'])[:-2] + [2] + list(measure_kwargs['bins'])[-2:])

        # print('subset', subset)
        # print('measure_kwargs', measure_kwargs)
        measure = self.measure_class(df=df_new, **measure_kwargs)


        result = measure.evaluate(["Yp"])
        del measure
        return result

    def name(self):
        return f"Surrogate-{self.measure_class.__name__}(mode={self.mode})[Xs replaced by Yp^S]"

# Measure for information theoretic measures based on distributions
class SurrogateMeasure_Y_by_hat_Y(Measure):
    def __init__(self, df, X, A, Y, measure_class: type, mode:  Literal["replace", "drop"], categorical_features = [], classifier = LogisticRegression(), split_ratio = 0.33, seed = 0, **kwargs):
        if 'fit_params' in kwargs:
            fit_params = kwargs['fit_params']
            kwargs.pop('fit_params')
        else:
            fit_params = {}
        self.classifier_base = ClassifierBase(classifier, df, features=X, sensitive_attribute=A, target_label=Y,
                                              categorical_features=categorical_features, split_ratio=split_ratio,
                                              seed=seed, **fit_params)
        self.X = X
        self.A = A
        self.Y = Y
        self.mode = mode
        self.measure_class = measure_class
        self.measure_kwargs = kwargs
        self.measure_kwargs['X'] = self.X
        self.measure_kwargs['A'] = self.A
        self.measure_kwargs['Y'] = self.Y





        df_new = self.classifier_base.classify_replace_mean(X) if self.mode == "replace" else self.classifier_base.classify_drop_out(X)
        # drop Y
        df_new = df_new.drop(columns=Y)

        self.Y = ['Yp']
        self.measure_kwargs['Y'] = self.Y

        # print('measure_kwargs', self.measure_kwargs)

        self.measure = self.measure_class(df=df_new, **self.measure_kwargs)

    def evaluate(self, Xs: list[str]) -> float:
        if len(Xs) == 0:
            return 0.0
        # print('subset', Xs)
        return self.measure.evaluate(Xs)

    def name(self):
        return f"Surrogate-{self.measure_class.__name__}(mode={self.mode})[Y replaced by Yp]"

class SurrogateMeasure_Y_by_hat_Y_Xs_hat_YS(Measure):
    def __init__(self, df, X, A, Y, measure_class: type, mode:  Literal["replace", "drop"], categorical_features = [], classifier = LogisticRegression(), split_ratio = 0.33, seed = 0, **kwargs):

        self.X = X
        self.A = A
        self.Y = Y
        self.header = X + A + Y
        if 'fit_params' in kwargs:
            fit_params = kwargs['fit_params']
            kwargs.pop('fit_params')
        else:
            fit_params = {}
        self.classifier_base = ClassifierBase(classifier, df, features=X, sensitive_attribute=A, target_label=Y,
                                              categorical_features=categorical_features, split_ratio=split_ratio,
                                              seed=seed, **fit_params)
        self.mode = mode
        self.measure_kwargs = kwargs
        self.measure_kwargs['X'] = self.X
        self.measure_kwargs['A'] = self.A
        self.measure_kwargs['Y'] = self.Y

        self.measure_class = measure_class
        df_new = self.classifier_base.classify_replace_mean(X) if mode == "replace" else self.classifier_base.classify_drop_out(X)
        # get Y_p
        self.Yp_df = df_new[['Yp']].copy()
        # rename Y_p to Y_p total
        self.Yp_df.rename(columns={'Yp': 'Yp_total'}, inplace=True)

    def evaluate(self, subset: list[str]) -> float:
        # print('subset', subset)
        if len(subset) == 0:
            return 0.0
        df_new = self.classifier_base.classify_replace_mean(
            subset) if self.mode == "replace" else self.classifier_base.classify_drop_out(subset)
        measure_kwargs = self.measure_kwargs.copy()
        indices_for_removal = []
        for index in range(len(self.measure_kwargs['X'])):
            if self.measure_kwargs['X'][index] in subset:
                indices_for_removal.append(index)
                # drop the column from the features to avoid redundancy
                df_new = df_new.drop(columns=[self.measure_kwargs['X'][index]])

        if 'bins' in measure_kwargs:
            # remove indices for removal from the bins
            measure_kwargs['bins'] = tuple(
                [b for i, b in enumerate(measure_kwargs['bins']) if i not in indices_for_removal])
        # remove subset from the features
        measure_kwargs['X'] = [x for x in self.measure_kwargs['X'] if x not in subset]
        # add 'Yp' to the features for evaluation
        measure_kwargs['X'] = measure_kwargs['X'] + ['Yp']  # Ensure 'Yp' is included in the features for evaluation
        # drop 'Y' from the features
        df_new = df_new.drop(columns=self.Y)
        # Add 'Yp_total' to the features
        df_new = pd.concat([df_new, self.Yp_df], axis=1)
        # rename 'Y' in measure_kwargs to 'Yp_total'
        measure_kwargs['Y'] = ['Yp_total']
        # Add bin for 'Yp'
        if 'bins' in self.measure_kwargs:
            measure_kwargs['bins'] = tuple(list(measure_kwargs['bins'])[:-2] + [2] + list(measure_kwargs['bins'])[-2:])
        # print('subset', subset)
        # print('measure_kwargs', measure_kwargs)
        measure = self.measure_class(df=df_new, **measure_kwargs)
        result = measure.evaluate(["Yp"])
        del measure
        return result

    def name(self):
        return f"Surrogate-{self.measure_class.__name__}(mode={self.mode})[Y replaced by Yp and Xs replaced by Yp^S]"