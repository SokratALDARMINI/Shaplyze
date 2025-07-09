from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
GAMMA = 0.5
class MMD:

    # @staticmethod
    # def compute_mmd(X0, X1, gamma=0.5, batch_size=1024):
    #     """
    #     Compute the MMD between two samples X0 and X1 using optional batch-wise kernel computation.
    #
    #     Parameters:
    #     - X0: First set of samples (pandas DataFrame or NumPy array).
    #     - X1: Second set of samples (pandas DataFrame or NumPy array).
    #     - gamma (float): Kernel bandwidth for the RBF kernel.
    #     - batch_size (int or None): Batch size for kernel computation. If None, use full batch.
    #
    #     Returns:
    #     - float: Square root of the MMD^2 value (i.e., the MMD).
    #     """
    #     # Convert to NumPy if necessary
    #     X = X0.to_numpy() if hasattr(X0, 'to_numpy') else X0
    #     Y = X1.to_numpy() if hasattr(X1, 'to_numpy') else X1
    #
    #     # Reshape to 2D if needed
    #     if X.ndim == 1:
    #         X = X.reshape(-1, 1)
    #     if Y.ndim == 1:
    #         Y = Y.reshape(-1, 1)
    #
    #     m, n = X.shape[0], Y.shape[0]
    #
    #     # Default to full-batch
    #     batch_size_X = m if batch_size is None else batch_size
    #     batch_size_Y = n if batch_size is None else batch_size
    #
    #     # Compute K_XX
    #     sum_K_XX = 0.0
    #     for i in range(0, m, batch_size_X):
    #         X_batch = X[i:i + batch_size_X]
    #         K_batch = rbf_kernel(X_batch, X, gamma=gamma)
    #         sum_K_XX += K_batch.sum()
    #
    #     # Compute K_YY
    #     sum_K_YY = 0.0
    #     for i in range(0, n, batch_size_Y):
    #         Y_batch = Y[i:i + batch_size_Y]
    #         K_batch = rbf_kernel(Y_batch, Y, gamma=gamma)
    #         sum_K_YY += K_batch.sum()
    #
    #     # Compute K_XY
    #     sum_K_XY = 0.0
    #     for i in range(0, m, batch_size_X):
    #         X_batch = X[i:i + batch_size_X]
    #         K_batch = rbf_kernel(X_batch, Y, gamma=gamma)
    #         sum_K_XY += K_batch.sum()
    #
    #     mmd2 = (sum_K_XX / (m * m)) + (sum_K_YY / (n * n)) - 2 * (sum_K_XY / (m * n))
    #     return np.sqrt(max(mmd2, 0.0))  # Safe for numerical precision
    @staticmethod
    def compute_mmd(X0, X1, gamma=0.5, batch_size=1024, threshold=50167):
        """
        Compute the MMD between X0 and X1 using optional sampling and batch-wise kernel computation.

        Parameters:
        - X0, X1: Input datasets (NumPy arrays or DataFrames)
        - gamma: RBF kernel bandwidth
        - batch_size: Batch size for kernel computation
        - sampling: If True and dataset > threshold, use a random fraction of data
        - fraction: Fraction of data to sample (default 0.33)
        - threshold: Dataset size above which sampling is applied

        Returns:
        - float: MMD value
        """
        X = X0.to_numpy() if hasattr(X0, 'to_numpy') else X0
        Y = X1.to_numpy() if hasattr(X1, 'to_numpy') else X1

        # Reshape to 2D if needed
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        m, n = X.shape[0], Y.shape[0]

        # Apply sampling if requested and data is large
        if (m > threshold or n > threshold):
            idx_X = np.random.choice(m, min(m,50167), replace=False)
            idx_Y = np.random.choice(n, min(m,50167), replace=False)
            X = X[idx_X]
            Y = Y[idx_Y]
            m, n = X.shape[0], Y.shape[0]

        # Set batch sizes
        batch_size_X = m if batch_size is None else batch_size
        batch_size_Y = n if batch_size is None else batch_size

        # Compute K_XX
        sum_K_XX = 0.0
        for i in range(0, m, batch_size_X):
            X_batch = X[i:i + batch_size_X]
            K_batch = rbf_kernel(X_batch, X, gamma=gamma)
            sum_K_XX += K_batch.sum()

        # Compute K_YY
        sum_K_YY = 0.0
        for i in range(0, n, batch_size_Y):
            Y_batch = Y[i:i + batch_size_Y]
            K_batch = rbf_kernel(Y_batch, Y, gamma=gamma)
            sum_K_YY += K_batch.sum()

        # Compute K_XY
        sum_K_XY = 0.0
        for i in range(0, m, batch_size_X):
            X_batch = X[i:i + batch_size_X]
            K_batch = rbf_kernel(X_batch, Y, gamma=gamma)
            sum_K_XY += K_batch.sum()

        mmd2 = (sum_K_XX / (m * m)) + (sum_K_YY / (n * n)) - 2 * (sum_K_XY / (m * n))
        return np.sqrt(max(mmd2, 0.0))  # Safe for numerical precision

    @staticmethod
    def mmd(df, Y, X, gamma=GAMMA):
        """
        Compute MMD between distributions of features X conditioned on binary target Y.

        Parameters:
        - df (pd.DataFrame): The dataset containing features and target.
        - X (list[str]): List of feature column names to use in MMD.
        - Y (list[str]): List with one string element: name of the binary target column.

        Returns:
        - float: MMD value between samples where Y == 0 and Y == 1.
        """
        # Split dataset by binary target values
        df_X1 = df[df[Y[0]] == 1][X]  # Features for Y=1
        df_X0 = df[df[Y[0]] == 0][X]  # Features for Y=0

        # Compute MMD between the two distributions
        mmd_value = MMD.compute_mmd(df_X1, df_X0, gamma=gamma)

        return mmd_value

    @staticmethod
    def cmmd(df, Z, A, B, gamma=GAMMA):
        """
        Compute Conditional MMD: CMMD(Z, A | B)
        Estimates E_B[ MMD(Z | B=b, A | B=b) ] over unique values of B.

        Parameters:
        - df (pd.DataFrame): Dataset containing features.
        - Z (list[str]): Target feature(s) (distribution 1).
        - A (list[str]): Second feature(s) (distribution 2).
        - B (list[str]): Conditioning feature(s) (usually 1 variable).
        - gamma (float): Kernel bandwidth.

        Returns:
        - float: Estimated conditional MMD.
        """
        assert len(B) == 1, "Currently supports single conditioning variable only."
        B_col = B[0]
        cmmd_total = 0.0

        for b_val in df[B_col].unique():
            df_b = df[df[B_col] == b_val]
            if len(df_b) < 2:
                continue  # skip if too few samples

            # Compute MMD between Z and A in the subpopulation where B = b_val
            mmd_b = MMD.mmd(df_b, Z, A, gamma=gamma)
            weight = len(df_b) / len(df)
            cmmd_total += weight * mmd_b

        return cmmd_total
