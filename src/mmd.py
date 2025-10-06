from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
GAMMA = 0.5
class MMD:

    @staticmethod
    def compute_mmd(X0, X1, gamma=0.5, batch_size=1024, threshold=51200):
        """ 300032
        Compute the MMD between X0 and X1 using optional sampling and full batch-wise kernel computation.

        Parameters:
        - X0, X1: Input datasets (NumPy arrays or DataFrames)
        - gamma: RBF kernel bandwidth
        - batch_size: Batch size for kernel computation
        - threshold: Dataset size above which sampling is applied

        Returns:
        - float: MMD value
        """

        # Convert to NumPy arrays
        X = X0.to_numpy() if hasattr(X0, 'to_numpy') else X0
        Y = X1.to_numpy() if hasattr(X1, 'to_numpy') else X1

        # Reshape to 2D if needed
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        m, n = X.shape[0], Y.shape[0]

        rng = np.random.default_rng(seed=42)  # Create a reproducible generator

        # Apply sampling if data is large
        if m > threshold:
            idx_X = rng.choice(m, threshold, replace=False)
            X = X[idx_X]
            m = threshold
        if n > threshold:
            idx_Y = rng.choice(n, threshold, replace=False)
            Y = Y[idx_Y]
            n = threshold

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

        # Compute K_XY with double batching
        sum_K_XY = 0.0
        for i in range(0, m, batch_size_X):
            X_batch = X[i:i + batch_size_X]
            # Compute kernel block against the full Y
            K_batch = rbf_kernel(X_batch, Y, gamma=gamma)
            sum_K_XY += K_batch.sum()

        # Compute MMD^2
        mmd2 = (sum_K_XX / (m * m)) + (sum_K_YY / (n * n)) - 2 * (sum_K_XY / (m * n))
        return np.sqrt(max(mmd2, 0.0))  # Ensure non-negative due to numerical precision

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
