"""
information_gaussian.py

Gaussian (differential) information-theoretic measures for Shaplyze.

Functions:
    mi(df, Z, A): Mutual information I(Z;A) under Gaussian assumption.
    entropy(df, Z): Differential entropy H(Z) for Gaussian variables.
    cond_entropy(df, Z, A): Conditional differential entropy H(Z|A).
    cmi(df, Z, A, B): Conditional mutual information I(Z;A|B).
"""
import numpy as np
import pandas as pd
from typing import List

class MIGaussian:

    @staticmethod
    def mi(df: pd.DataFrame, Z, A, eps=1e-8):
        """
        Estimate I(A;Z) for multivariate Gaussian (in bits) with ridging on both covariances.
        """
        data_Z = df[Z].values
        data_A = df[A].values
        if data_A.size == 0 or data_Z.size == 0:
            raise ValueError("At least one feature must be provided for both A and Z")

        # ML covariances
        Σ_A = np.cov(data_A, rowvar=False, ddof=0)
        Σ_Z = np.cov(data_Z, rowvar=False, ddof=0)
        # ensure 2D
        if Σ_A.ndim == 0: Σ_A = Σ_A.reshape(1,1)
        if Σ_Z.ndim == 0: Σ_Z = Σ_Z.reshape(1,1)

        # add ridge to both
        Σ_A += eps * np.eye(Σ_A.shape[0])
        Σ_Z += eps * np.eye(Σ_Z.shape[0])

        # cross‐covariance
        joint = np.cov(np.hstack([data_A, data_Z]), rowvar=False, ddof=0)
        p = data_A.shape[1]
        Σ_AZ = joint[:p, p:]

        # conditional covariance: Σ_{A|Z} = Σ_A − Σ_AZ Σ_Z^{-1} Σ_AZ^T
        inv_term = np.linalg.solve(Σ_Z, Σ_AZ.T)
        Σ_cond = Σ_A - Σ_AZ @ inv_term
        if Σ_cond.ndim == 0: Σ_cond = Σ_cond.reshape(1,1)

        # log‐determinants
        sA, ldA = np.linalg.slogdet(Σ_A)
        sc, ldc = np.linalg.slogdet(Σ_cond)
        if sA <= 0 or sc <= 0:
            raise ValueError("Covariance still not positive definite after ridging.")

        # MI in bits
        return 0.5 * (ldA - ldc) / np.log(2)

    @staticmethod
    def entropy(df: pd.DataFrame, Z):
        """
        Differential entropy h(Z) for a multivariate Gaussian Z (in bits):
          h(Z) = ½ [ d·ln(2πe) + ln det Σ_Z ] / ln 2
        """
        if not Z:
            return 0.0

        X = df[Z].values
        Σ = np.cov(X, rowvar=False, ddof=0)
        # ensure 2D
        if Σ.ndim == 0:
            Σ = Σ.reshape(1,1)

        sign, logdet = np.linalg.slogdet(Σ)
        if sign <= 0:
            raise ValueError("Covariance of Z is not positive definite.")

        d = Σ.shape[0]
        # differential entropy in nats: ½[d·(1 + ln(2π)) + ln det Σ]
        h_nats = 0.5 * (d * (1 + np.log(2*np.pi)) + logdet)
        return h_nats / np.log(2)  # convert to bits

    @staticmethod
    def cond_entropy(df: pd.DataFrame, Z, A):
        """
        Conditional differential entropy h(Z|A) for Gaussians (in bits):
          h(Z|A) = h(Z ∪ A) - h(A)
        """
        if not Z:
            return 0.0
        if not A:
            return MIGaussian.entropy(df, Z)
        ZA = Z + [a for a in A if a not in Z]  # ensure no duplicates
        return MIGaussian.entropy(df, ZA) - MIGaussian.entropy(df, A)


    @staticmethod
    def cmi(df: pd.DataFrame, Z, A, B, eps=1e-8):
        """
        Conditional mutual information I(Z;A|B) in bits:
          I(Z;A|B) = I(Z;A,B) - I(Z;B)
        """
        if not Z or not A:
            return 0.0

        # ensure no duplicates when we form A∪B
        AB = A + [b for b in B if b not in A]

        return MIGaussian.mi(df, Z, AB, eps=eps) - MIGaussian.mi(df, Z, B, eps=eps)

    @staticmethod
    def uni(df: pd.DataFrame, Z, A, B, eps=1e-8):
        """
        Unique information Uni(Z:A|B) for Gaussian variables (MMI PID definition):
            Uni(Z:A|B) = I(Z;A) - Red(Z:(A,B))
        where Red(Z:(A,B)) = min(I(Z;A), I(Z;B)).

        Parameters
        ----------
        df : pd.DataFrame
            Data containing columns for variables Z, A, and B.
        Z : List[str]
            List of column names for target variables.
        A : List[str]
            List of column names for first source variables.
        B : List[str]
            List of column names for second source variables.

        Returns
        -------
        float
            The unique information component in bits.
        """
        iZA = MIGaussian.mi(df, Z, A, eps= eps)
        iZB = MIGaussian.mi(df, Z, B, eps= eps)
        red = min(iZA, iZB)
        return iZA - red

    @staticmethod
    def red(df: pd.DataFrame, Z, A, B, eps=1e-8):
        """
        Redundant information Red(Z:(A,B)) for Gaussian variables (MMI PID):
            Red(Z:(A,B)) = min(I(Z;A), I(Z;B)).

        Parameters
        ----------
        df : pd.DataFrame
        Z : List[str]
        A : List[str]
        B : List[str]

        Returns
        -------
        float
            The redundant information component in bits.
        """
        iZA = MIGaussian.mi(df, Z, A, eps= eps)
        iZB = MIGaussian.mi(df, Z, B, eps= eps)
        return min(iZA, iZB)

    @staticmethod
    def syn(df: pd.DataFrame, Z, A, B, eps=1e-8):
        """
        Synergistic information Syn(Z:A|B) for Gaussian variables (MMI PID):
            Syn(Z:A|B) = I(Z;A|B) - Uni(Z:A|B).

        Parameters
        ----------
        df : pd.DataFrame
        Z : List[str]
        A : List[str]
        B : List[str]

        Returns
        -------
        float
            The synergistic information component in bits.
        """
        return MIGaussian.cmi(df, Z, A, B, eps= eps) - MIGaussian.uni(df, Z, A, B, eps= eps)

