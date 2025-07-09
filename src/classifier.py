import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

class ClassifierBase:
    def __init__(self, classifier, df, features, sensitive_attribute, target_label, categorical_features = [], split_ratio = 0.33, seed = 0):
        """
       A utility class for training and evaluating classifiers with feature perturbation
       via mean-replacement or feature-dropout, especially useful for fairness and attribution analysis.

       This class preprocesses categorical and numerical features using one-hot encoding and standardization,
       trains two separate clones of the given classifier for mean-replacement and dropout strategies, and
       provides predictions with feature modifications to support fairness evaluations.

       Attributes:
           classifier_drop: sklearn classifier used for feature drop-out strategy.
           classifier_replace: sklearn classifier used for feature mean-replacement strategy.
           A (pd.Series): Sensitive attribute column.
           Y (pd.Series): Target label column.
           X (pd.DataFrame): Raw feature matrix (before transformation).
           features (list[str]): List of feature column names.
           sensitive_attribute (list[str]): Name of the sensitive attribute.
           target_label (list[str]): Name of the target label column.
           categorical_features (list[str]): Names of categorical features.
           cat_features (list[str]): Subset of features that are categorical.
           non_cat_features (list[str]): Subset of features that are not categorical.
           X_transformed (pd.DataFrame): Transformed features after encoding and scaling.
           feature_replacement_map (dict): Maps categorical feature names to one-hot columns.
           seed (int): Random seed used for train-test split.
           split_ratio (float): Proportion of data used for testing.
           X_train, X_test (pd.DataFrame): Transformed training and testing features.
           y_train, y_test (pd.Series): Training and testing labels.
           A_train, A_test (pd.Series): Sensitive attributes for training and testing data.
           mean_dic (dict): Dictionary of mean values for all transformed features.
           X_test_original (pd.DataFrame): Untouched test features (raw, before encoding).
       """

        self.classifier_drop = clone(classifier)
        self.classifier_replace = clone(classifier)

        self.A = df[sensitive_attribute].copy(deep=True)
        self.Y =  df[target_label].copy(deep=True)
        self.X =  df[features].copy(deep=True)

        self.features = features
        self.sensitive_attribute = sensitive_attribute
        self.target_label = target_label

        self.categorical_features = categorical_features

        self.cat_features = [f for f in features if f in categorical_features]
        self.non_cat_features = [f for f in features if f not in categorical_features]

        one_hot_encoder = OneHotEncoder(sparse_output=False, drop=None)
        if len(self.cat_features)>0:
            cat_encoded = one_hot_encoder.fit_transform(self.X[self.cat_features])
            cat_feature_names = one_hot_encoder.get_feature_names_out(self.cat_features)
            cat_encoded_df = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=self.X.index)
        else:
            cat_encoded_df = pd.DataFrame(index=self.X.index)

        self.X = self.X.drop(columns=self.cat_features)

        if len(self.non_cat_features) > 0:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.X[self.non_cat_features])
            non_cat_scaled_df = pd.DataFrame(X_scaled, columns=self.non_cat_features, index=self.X.index)
        else:
            non_cat_scaled_df = pd.DataFrame(index=self.X.index)

        self.X_transformed = pd.concat([cat_encoded_df, non_cat_scaled_df], axis=1)
        self.feature_replacement_map = {cat: [col for col in cat_feature_names if col.startswith(cat + "_")] for cat in self.cat_features}

        self.seed = seed
        self.split_ratio = split_ratio
        # First: split indices once
        X_train_original, X_test_original, y_train_original, y_test_original, A_train_original, A_test_original = train_test_split(
            df[features], df[target_label], df[sensitive_attribute],
            test_size=self.split_ratio, random_state=self.seed
        )

        self.X_test_original = X_test_original
        self.X_train_original = X_train_original
        # Then: align transformed features with these same indices
        self.X_train = self.X_transformed.loc[X_train_original.index]
        self.X_test = self.X_transformed.loc[X_test_original.index]
        self.y_train = y_train_original
        self.y_test = y_test_original
        self.A_train = A_train_original
        self.A_test = A_test_original

        self.mean_dic = self.X_transformed.mean().to_dict()
        self.classifier_replace.fit(self.X_train, self.y_train.values.ravel())

        # self.ref_acc = np.min(self.Y[0].mean(), 1 - self.Y[0].mean())
        # self.ref_dis_DP = 0
        # self.ref_dis_EO = 0
        # self.ref_dis_EOP_0 = 0
        # self.ref_dis_EOP_1 = 0


    def classify_replace_mean(self, features_subset):
        """
        Predict using a model trained on all features, but during inference, replace
        excluded features with their global mean (or one-hot mean for categorical).

        Args:
            features_subset (list[str]): Subset of features to retain during prediction.

        Returns:
            pd.DataFrame: DataFrame including original test features, sensitive attribute,
                          true label, and predicted label ('Yp').
        """


        X_test = self.X_test.copy(deep=True)

        y_test = self.y_test.values.ravel()  # Convert y_test to 1D array

        A_test = self.A_test.values.ravel()  # Convert y_test to 1D array

        # replace each feature not in "features" with the mean of the feature
        for feature in self.X:
            if feature not in features_subset:
                if feature in self.cat_features:
                    for col in self.feature_replacement_map[feature]:
                        X_test[col] = self.mean_dic[col]
                else:
                        X_test[feature] = self.mean_dic[feature]

        # Prediction
        y_pred = self.classifier_replace.predict(X_test)
        # create a dataframe with the features and A and Y and Yp
        df = self.X_test_original
        df[self.sensitive_attribute[0]] = A_test
        df[self.target_label[0]] = y_test
        df['Yp'] = y_pred

        return df

    def classify_drop_out(self, features_subset):
        """
        Retrain the model after permanently dropping the excluded features from training and test data.

        Args:
            features_subset (list[str]): Subset of features to retain (others are dropped).

        Returns:
            pd.DataFrame: DataFrame including original test features, sensitive attribute,
                          true label, and predicted label ('Yp').
        """

        X_test = self.X_test.copy(deep=True)
        X_train = self.X_train.copy(deep=True)

        y_test = self.y_test.values.ravel()  # Convert y_test to 1D array

        A_test = self.A_test.values.ravel()  # Convert y_test to 1D array

        # drop out features not in "features" vector all at once
        for feature in self.X:
            if feature not in features_subset:
                if feature in self.cat_features:
                    for col in self.feature_replacement_map[feature]:
                        X_test.drop(columns=col, inplace=True)
                        X_train.drop(columns=col, inplace=True)
                else:
                    X_test.drop(columns=feature, inplace=True)
                    X_train.drop(columns=feature, inplace=True)


        if X_train.empty:
            df = self.X_test_original
            df[self.sensitive_attribute[0]] = A_test
            df[self.target_label[0]] = y_test
            # Yp is 1 if y_train mean on the training datasett is greater than 0.5, else 0
            df['Yp'] = int((self.y_train.mean() > 0.5).item())

            return df

        self.classifier_drop.fit(X_train, self.y_train.values.ravel())

        # predict the target variable using the classifier
        # Prediction
        y_pred = self.classifier_drop.predict(X_test)
        # create a dataframe with the features and A and Y and Yp
        df = self.X_test_original
        df[self.sensitive_attribute[0]] = A_test
        df[self.target_label[0]] = y_test
        df['Yp'] = y_pred

        return df





