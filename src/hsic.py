import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
# EPS = 1e-6
# GAMMA_X = 1.0
# GAMMA_Y = 1.0
class HSIC:

    @staticmethod
    def hsic(df, X, Y, gamma_X=1.0, gamma_Y=1.0, batch_size=1024, seed=0):
        """
        Compute the Hilbert-Schmidt Independence Criterion (HSIC) between X and Y.

        Parameters:
        - X: np.ndarray of shape (n_samples, n_features_X)
        - Y: np.ndarray of shape (n_samples, n_features_Y)
        - gamma_X: float, RBF kernel bandwidth for X
        - gamma_Y: float, RBF kernel bandwidth for Y
        - batch_size: int or None. If None, computes full HSIC. If int, uses mini-batches of this size.
        - seed: int or None. For reproducibility in batching.

        Returns:
        - float: estimated HSIC value
        """
        if len(X) == 0 or len(Y) == 0:
            return 0.0
        X = df[X].values
        Y = df[Y].values
        def _hsic_core(Xb, Yb, gamma_X, gamma_Y):
            n = Xb.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            Kx = rbf_kernel(Xb, Xb, gamma=gamma_X)
            Ky = rbf_kernel(Yb, Yb, gamma=gamma_Y)
            Kxc = H @ Kx @ H
            Kyc = H @ Ky @ H
            return np.trace(Kxc @ Kyc) / ((n - 1) ** 2)

        if batch_size is None or batch_size >= X.shape[0]:
            return _hsic_core(X, Y, gamma_X, gamma_Y)

        # Mini-batch mode
        if seed is not None:
            np.random.seed(seed)

        n = X.shape[0]
        indices = np.random.permutation(n)
        hsic_vals = []
        for i in range(0, n - batch_size + 1, batch_size):
            idx = indices[i:i + batch_size]
            hsic_vals.append(_hsic_core(X[idx], Y[idx], gamma_X, gamma_Y))
        return np.mean(hsic_vals)

    @staticmethod
    def chsic(df, X_cols, Y_cols, Z_cols, gamma_X=1.0, gamma_Y=1.0, batch_size=1024, seed=0):
        """
        Estimate conditional HSIC(X, Y | Z) using stratification over unique values of Z.

        Parameters:
        - df (pd.DataFrame): Input dataset
        - X_cols (list[str]): Feature columns for X
        - Y_cols (list[str]): Feature columns for Y
        - Z_cols (list[str]): Conditioning variable(s), assumed categorical or discrete
        - gamma_X (float): RBF kernel bandwidth for X
        - gamma_Y (float): RBF kernel bandwidth for Y
        - batch_size (int or None): Batch size for HSIC estimation
        - seed (int): Random seed for batching

        Returns:
        - float: Estimated conditional HSIC
        """
        assert len(Z_cols) == 1, "Currently supports a single conditioning variable only."
        Z_col = Z_cols[0]
        chsic_total = 0.0

        for z_val in df[Z_col].unique():
            df_z = df[df[Z_col] == z_val]
            if len(df_z) < 2:
                continue  # skip subsets too small

            hsic_val = HSIC.hsic(df_z, X_cols, Y_cols, gamma_X=gamma_X, gamma_Y=gamma_Y, batch_size=batch_size, seed=seed)
            weight = len(df_z) / len(df)
            chsic_total += weight * hsic_val

        return chsic_total

    @staticmethod
    def nocco(df, X, Y, gamma_X=1.0, gamma_Y=1.0, eps=1e-6, batch_size=1024, seed=0):
        """
        Compute the NOCCO (Normalized Cross-Covariance Operator) between X and Y.

        Parameters:
        - X: np.ndarray of shape (n_samples, n_features_X)
        - Y: np.ndarray of shape (n_samples, n_features_Y)
        - gamma_X: float, RBF kernel bandwidth for X
        - gamma_Y: float, RBF kernel bandwidth for Y
        - eps: float, regularization constant
        - batch_size: int or None. If None, uses full data. Else, average over random batches.
        - seed: int or None. Random seed for reproducibility.

        Returns:
        - float: NOCCO value (between 0 and 1)
        """

        if len(X) == 0 or len(Y) == 0:
            return 0.0
        X = df[X].values
        Y = df[Y].values

        def _nocco_core(Xb, Yb, gamma_X, gamma_Y, eps):
            n = Xb.shape[0]
            I = np.eye(n)
            H = I - np.ones((n, n)) / n
            Kx = rbf_kernel(Xb, Xb, gamma=gamma_X)
            Ky = rbf_kernel(Yb, Yb, gamma=gamma_Y)
            HKxH = H @ Kx @ H
            HKyH = H @ Ky @ H
            reg_x = HKxH + n * eps * I
            reg_y = HKyH + n * eps * I
            Rx = HKxH @ np.linalg.inv(reg_x)
            Ry = HKyH @ np.linalg.inv(reg_y)
            return np.trace(Rx @ Ry)

        if batch_size is None or batch_size >= X.shape[0]:
            return _nocco_core(X, Y, gamma_X, gamma_Y, eps)

        # Mini-batch mode
        if seed is not None:
            np.random.seed(seed)

        n = X.shape[0]
        indices = np.random.permutation(n)
        nocco_vals = []
        for i in range(0, n - batch_size + 1, batch_size):
            idx = indices[i:i + batch_size]
            nocco_vals.append(_nocco_core(X[idx], Y[idx], gamma_X, gamma_Y, eps))
        return np.mean(nocco_vals)

    @staticmethod
    def cnocco(df, X_cols, Y_cols, Z_cols, gamma_X=1.0, gamma_Y=1.0, eps=1e-6, batch_size=128, seed=0):
        """
        Estimate conditional NOCCO(X, Y | Z) by stratifying over discrete values of Z.

        Parameters:
        - df (pd.DataFrame): Input dataset
        - X_cols (list[str]): Feature columns for X
        - Y_cols (list[str]): Feature columns for Y
        - Z_cols (list[str]): Conditioning variable(s), assumed categorical or discrete
        - gamma_X (float): RBF kernel bandwidth for X
        - gamma_Y (float): RBF kernel bandwidth for Y
        - eps (float): Regularization parameter for NOCCO
        - batch_size (int or None): Mini-batch size for NOCCO estimation
        - seed (int): Random seed for batching

        Returns:
        - float: Estimated conditional NOCCO
        """
        assert len(Z_cols) == 1, "Currently only supports a single conditioning variable."
        Z_col = Z_cols[0]
        cnocco_total = 0.0

        for z_val in df[Z_col].unique():
            df_z = df[df[Z_col] == z_val]
            if len(df_z) < 2:
                continue  # skip if subset too small

            # X = df_z[X_cols].to_numpy()
            # Y = df_z[Y_cols].to_numpy()
            val = HSIC.nocco(df_z, X_cols, Y_cols, gamma_X=gamma_X, gamma_Y=gamma_Y, eps=eps, batch_size=batch_size, seed=seed)
            weight = len(df_z) / len(df)
            cnocco_total += weight * val

        return cnocco_total
