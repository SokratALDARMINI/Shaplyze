import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import copy
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

class ClassifierBase:
    def __init__(self, classifier, df, features, sensitive_attribute, target_label, categorical_features = [], split_ratio = 0.33, seed = 0, **fit_params):
        """
        A class for training and evaluating classifiers with feature intervention
        via mean-replacement or feature-dropout.

        Parameters:
            classifier (object): A scikit-learn–compatible classifier (or similar)
                to be cloned for training under different strategies.
            df (pd.DataFrame): Input dataset containing features, sensitive attribute,
                and target label.
            features (list[str]): Column names of input features used for training.
            sensitive_attribute (list[str]): Column name(s) of the sensitive attribute.
                Typically a single-element list.
            target_label (list[str]): Column name(s) of the target variable.
                Typically a single-element list.
            categorical_features (list[str], optional): Subset of `features` that
                are categorical and should be one-hot encoded. Defaults to [].
            split_ratio (float, optional): Proportion of the dataset reserved for testing.
                Defaults to 0.33.
            seed (int, optional): Random seed for train–test splitting. Defaults to 0.
            **fit_params: Additional keyword arguments passed to the classifier’s
                `.fit()` method.

        Attributes:
            classifier_drop (object): Classifier instance used for feature drop-out strategy.
            classifier_replace (object): Classifier instance used for feature mean-replacement strategy.
            A (pd.Series): Sensitive attribute column.
            Y (pd.Series): Target label column.
            X (pd.DataFrame): Raw feature matrix (before transformation).
            cat_features (list[str]): Categorical features among `features`.
            non_cat_features (list[str]): Non-categorical features among `features`.
            X_transformed (pd.DataFrame): Features after one-hot encoding and scaling.
            feature_replacement_map (dict): Mapping from each categorical feature to its one-hot encoded columns.
            X_train, X_test (pd.DataFrame): Transformed training and testing feature sets.
            y_train, y_test (pd.Series): Training and testing target labels.
            A_train, A_test (pd.Series): Sensitive attributes for training and testing sets.
            X_train_original, X_test_original (pd.DataFrame): Untransformed (raw) features for train/test split.
            mean_dic (dict): Dictionary of mean values for all transformed features.
        """

        self.classifier_drop = self._clone_model(classifier) # Model for feature drop-out strategy
        self.classifier_replace = self._clone_model(classifier) # Model for mean-replacement strategy

        # Extract sensitive attribute, target label, and features from DataFrame
        self.A = df[sensitive_attribute].copy(deep=True)
        self.Y =  df[target_label].copy(deep=True)
        self.X =  df[features].copy(deep=True)

        # Store feature metadata
        self.features = features
        self.sensitive_attribute = sensitive_attribute
        self.target_label = target_label

        self.categorical_features = categorical_features

        # Split into categorical vs. non-categorical features
        self.cat_features = [f for f in features if f in categorical_features]
        self.non_cat_features = [f for f in features if f not in categorical_features]
        self.fit_params = fit_params

        # One-hot encode categorical features
        one_hot_encoder = OneHotEncoder(sparse_output=False, drop=None)
        if len(self.cat_features)>0:
            cat_encoded = one_hot_encoder.fit_transform(self.X[self.cat_features])
            cat_feature_names = one_hot_encoder.get_feature_names_out(self.cat_features)
            cat_encoded_df = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=self.X.index)
        else:
            cat_encoded_df = pd.DataFrame(index=self.X.index)

        # Drop original categorical columns
        self.X = self.X.drop(columns=self.cat_features)

        # Standardize non-categorical features
        if len(self.non_cat_features) > 0:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.X[self.non_cat_features])
            non_cat_scaled_df = pd.DataFrame(X_scaled, columns=self.non_cat_features, index=self.X.index)
        else:
            non_cat_scaled_df = pd.DataFrame(index=self.X.index)

        # Combine transformed categorical and numerical features
        self.X_transformed = pd.concat([cat_encoded_df, non_cat_scaled_df], axis=1)
        # Map original categorical feature → its one-hot encoded columns
        self.feature_replacement_map = {cat: [col for col in cat_feature_names if col.startswith(cat + "_")] for cat in self.cat_features}

        self.seed = seed
        self.split_ratio = split_ratio

        # Train–test split (using raw features, target, and sensitive attribute)
        X_train_original, X_test_original, y_train_original, y_test_original, A_train_original, A_test_original = train_test_split(
            df[features], df[target_label], df[sensitive_attribute],
            test_size=self.split_ratio, random_state=self.seed
        )

        # Save original (raw) train/test splits
        self.X_test_original = X_test_original
        self.X_train_original = X_train_original
        # Align transformed features with original split indices
        self.X_train = self.X_transformed.loc[X_train_original.index]
        self.X_test = self.X_transformed.loc[X_test_original.index]
        self.y_train = y_train_original
        self.y_test = y_test_original
        self.A_train = A_train_original
        self.A_test = A_test_original

        # Mean values of transformed features (used for mean-replacement strategy)
        self.mean_dic = self.X_transformed.mean().to_dict()

        # Train replacement-based model on full feature set
        if self.fit_params:
            self.classifier_replace.fit(self.X_train, self.y_train.values.ravel(), **self.fit_params)
        else:
            self.classifier_replace.fit(self.X_train, self.y_train.values.ravel())



    def _clone_model(self,classifier):
        # sklearn estimators have get_params
        if hasattr(classifier, "get_params"):
            return clone(classifier)
        else:
            # fallback for keras (or others): use deepcopy or rebuild
            return copy.deepcopy(classifier)

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

        # Copy transformed test set
        X_test = self.X_test.copy(deep=True)

        # Flatten y_test and A_test arrays
        y_test = self.y_test.values.ravel()  # Convert y_test to 1D array
        A_test = self.A_test.values.ravel()  # Convert y_test to 1D array

        # Replace excluded features with their mean value
        for feature in self.features:
            if feature not in features_subset:
                if feature in self.cat_features:
                    # Replace all one-hot columns of this categorical feature
                    for col in self.feature_replacement_map[feature]:
                        X_test[col] = self.mean_dic[col]
                else:
                        X_test[feature] = self.mean_dic[feature]

        # Predict with replacement-trained model
        y_pred = self.classifier_replace.predict(X_test)

        # Handle probability output (e.g., from keras) → binary labels
        if y_pred.ndim > 1:
            y_pred = (y_pred.ravel() > 0.5).astype(int)
        # Return DataFrame with original test features + A, Y, Yp
        df = self.X_test_original
        df[self.sensitive_attribute[0]] = A_test
        df[self.target_label[0]] = y_test
        df['Yp'] = y_pred

        return df

    def classify_drop_out(self, features_subset):
        """
            Retrain the model after permanently dropping the excluded features
            from training and test data.

            Args:
                features_subset (list[str]): Subset of features to retain (others are dropped).

            Returns:
                pd.DataFrame: DataFrame including original test features, sensitive attribute,
                              true label, and predicted label ('Yp').
        """
        # Copy transformed train/test sets
        X_test = self.X_test.copy(deep=True)
        X_train = self.X_train.copy(deep=True)

        # Flatten y_test and A_test arrays
        y_test = self.y_test.values.ravel()
        A_test = self.A_test.values.ravel()

        # Drop columns of features not in the subset
        for feature in self.X:
            if feature not in features_subset:
                if feature in self.cat_features:
                    # Drop all one-hot columns of this categorical feature
                    for col in self.feature_replacement_map[feature]:
                        X_test.drop(columns=col, inplace=True)
                        X_train.drop(columns=col, inplace=True)
                else:
                    X_test.drop(columns=feature, inplace=True)
                    X_train.drop(columns=feature, inplace=True)

        # Special case: no features left → predict majority class
        if X_train.empty:
            df = self.X_test_original
            df[self.sensitive_attribute[0]] = A_test
            df[self.target_label[0]] = y_test
            df['Yp'] = int((self.y_train.mean() > 0.5).item())
            return df
        # Retrain model on reduced feature set
        if self.fit_params:
            self.classifier_drop.fit(X_train, self.y_train.values.ravel(), **self.fit_params)
        else:
            self.classifier_drop.fit(X_train, self.y_train.values.ravel())

        # Predict on reduced test set
        y_pred = self.classifier_drop.predict(X_test)

        # Handle probability output (e.g., from keras) → binary labels
        if y_pred.ndim > 1:
            y_pred = (y_pred.ravel() > 0.5).astype(int)
        # Return DataFrame with original test features + A, Y, Yp
        df = self.X_test_original
        df[self.sensitive_attribute[0]] = A_test
        df[self.target_label[0]] = y_test
        df['Yp'] = y_pred

        return df





