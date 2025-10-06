import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

class HSIC:
    """
    A class implementing dependence measures based on kernels:
    - HSIC (Hilbert–Schmidt Independence Criterion)
    - Conditional HSIC
    - NOCCO (Normalized Cross-Covariance Operator)
    - Conditional NOCCO

    Notes:
        - Inputs (X, Y, Z) are lists of feature names from a DataFrame.
        - HSIC is computed using row-wise kernel interactions (trace and centering).
        - NOCCO is computed via normalized kernel cross-covariance and averaged across batches.
    """
    def hsic(df, X, Y, gamma_X=1.0, gamma_Y=1.0, batch_size=1024, threshold=51200, seed=0):
        """
        Memory-efficient HSIC(X, Y) between sets of features using RBF kernels.

        Parameters:
        - df (pd.DataFrame): Input dataset
        - X (list[str]): Feature columns for X
        - Y (list[str]): Feature columns for Y
        - gamma_X (float): Bandwidth parameter for X RBF kernel
        - gamma_Y (float): Bandwidth parameter for Y RBF kernel
        - batch_size (int): Block size for kernel computation
        - threshold (int): Maximum number of samples; above this, subsample without replacement
        - seed (int): Random seed for reproducibility

        Returns:
        - float: HSIC value
        """
        # Extract feature arrays
        X = df[X].values
        Y = df[Y].values
        n = X.shape[0]

        if n == 0:
            return 0.0

        # Subsample if dataset is too large
        if n > threshold:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, threshold, replace=False)
            X, Y = X[idx], Y[idx]
            n = threshold

        ones = np.ones((n, 1), dtype=np.float64)

        # Accumulators for HSIC components
        trace_KxKy = 0.0
        sum_Kx = 0.0
        sum_Ky = 0.0
        onesT_KxKy_ones = 0.0

        # Compute kernels blockwise for memory efficiency
        for i in range(0, n, batch_size):
            Xi = X[i:i + batch_size]
            Yi = Y[i:i + batch_size]
            for j in range(0, n, batch_size):
                Xj = X[j:j + batch_size]
                Yj = Y[j:j + batch_size]

                Kx_block = rbf_kernel(Xi, Xj, gamma=gamma_X)
                Ky_block = rbf_kernel(Yi, Yj, gamma=gamma_Y)

                # 1. Contribution to trace(Kx Ky)
                trace_KxKy += np.sum(Kx_block * Ky_block)

                # 2. Contribution to sum(Kx) and sum(Ky)
                sum_Kx += Kx_block.sum()
                sum_Ky += Ky_block.sum()

                # 3. Contribution to 1^T Kx Ky 1
                # Compute (Kx_block @ ones_j) and (Ky_block @ ones_j)
                ones_j = np.ones((Xj.shape[0], 1), dtype=np.float64)
                v1 = Kx_block @ ones_j
                v2 = Ky_block @ ones_j
                onesT_KxKy_ones += float((v1.T @ v2))

        # Combine into HSIC formula
        hsic_val = (trace_KxKy
                    + (sum_Kx * sum_Ky) / (n ** 2)
                    - (2.0 / n) * onesT_KxKy_ones) / ((n - 1) ** 2)

        return float(hsic_val)

    @staticmethod
    def hsic_old(df, X, Y, gamma_X=1.0, gamma_Y=1.0, batch_size=1024, seed=0):
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
        Conditional HSIC(X, Y | Z), stratified over discrete/categorical Z.

        Parameters:
        - df (pd.DataFrame): Input dataset
        - X_cols (list[str]): Feature columns for X
        - Y_cols (list[str]): Feature columns for Y
        - Z_cols (list[str]): Conditioning variable(s), assumed discrete
        - gamma_X, gamma_Y (float): Kernel bandwidths
        - batch_size (int): Batch size for HSIC estimation
        - seed (int): Random seed

        Returns:
        - float: Weighted average conditional HSIC
        """
        assert len(Z_cols) == 1, "Currently supports a single conditioning variable only."
        Z_col = Z_cols[0]
        chsic_total = 0.0

        # Stratify over each Z value
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
        Compute NOCCO(X, Y): Normalized Cross-Covariance Operator.

        Parameters:
        - df (pd.DataFrame): Input dataset
        - X (list[str]): Feature columns for X
        - Y (list[str]): Feature columns for Y
        - gamma_X, gamma_Y (float): Kernel bandwidths
        - eps (float): Regularization parameter
        - batch_size (int or None): If None, compute full NOCCO; else average across batches
        - seed (int): Random seed for batching

        Returns:
        - float: NOCCO value in [0, 1]
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

        if seed is not None:
            np.random.seed(seed)

        n = X.shape[0]
        indices = np.random.permutation(n)
        nocco_vals = []
        # Average across mini-batches
        for i in range(0, n - batch_size + 1, batch_size):
            idx = indices[i:i + batch_size]
            nocco_vals.append(_nocco_core(X[idx], Y[idx], gamma_X, gamma_Y, eps))
        return np.mean(nocco_vals)

    @staticmethod
    def cnocco(df, X_cols, Y_cols, Z_cols, gamma_X=1.0, gamma_Y=1.0, eps=1e-6, batch_size=128, seed=0):
        """
        Conditional NOCCO(X, Y | Z), stratified over discrete/categorical Z.

        Parameters:
        - df (pd.DataFrame): Input dataset
        - X_cols (list[str]): Feature columns for X
        - Y_cols (list[str]): Feature columns for Y
        - Z_cols (list[str]): Conditioning variable(s), assumed discrete
        - gamma_X, gamma_Y (float): Kernel bandwidths
        - eps (float): Regularization for NOCCO
        - batch_size (int): Batch size for NOCCO estimation
        - seed (int): Random seed for batching

        Returns:
        - float: Weighted average conditional NOCCO
        """
        assert len(Z_cols) == 1, "Currently only supports a single conditioning variable."
        Z_col = Z_cols[0]
        cnocco_total = 0.0

        # Stratify over each Z value
        for z_val in df[Z_col].unique():
            df_z = df[df[Z_col] == z_val]
            if len(df_z) < 2:
                continue  # skip if subset too small

            val = HSIC.nocco(df_z, X_cols, Y_cols, gamma_X=gamma_X, gamma_Y=gamma_Y, eps=eps, batch_size=batch_size, seed=seed)
            weight = len(df_z) / len(df)
            cnocco_total += weight * val

        return cnocco_total
