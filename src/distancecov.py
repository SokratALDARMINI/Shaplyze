

class DistanceCovariance:
    """
    A class to compute distance covariance and conditional distance covariance
    between random variables represented in a dit.Distribution.
    Inputs (Z, A, B) are specified as feature-name lists that get mapped to indices
    using the `header`.
    """

    @staticmethod
    def distanceCovCore(dist, Z, A):
        """
        Core routine to compute squared L² divergence between p(Z, A) and p(Z)p(A).

        Parameters:
        - dist (dit.Distribution): Distribution over the joint space of (Z, A).
        - Z (str): Label/name of the first coalesced variable (e.g., 'Z').
        - A (str): Label/name of the second coalesced variable (e.g., 'A').

        Returns:
        - float: Squared L² divergence between joint p(Z, A) and the product p(Z)p(A).
        """

        dist.make_dense()
        dist_Z = dist.coalesce(Z)  # Marginal distribution over Z
        dist_A = dist.coalesce(A)  # Marginal distribution over A

        dist_ZA_pmf = dist.pmf
        dist_Z_pmf = dist_Z.pmf
        dist_A_pmf = dist_A.pmf

        dist_Z_cordinates = [i for i in range(len(dist_Z.outcomes))]
        dist_A_cordinates = [i for i in range(len(dist_A.outcomes))]
        dist_ZA_cordinates = [i for i in range(len(dist.outcomes))]
        l2 = 0.0
        for zi in dist_Z_cordinates:
            for ai in dist_A_cordinates:
                # Joint probability at (Z=zi, A=ai)
                p_za = dist_ZA_pmf[zi * len(dist_A_cordinates) + ai]
                # Marginals
                p_z = dist_Z_pmf[zi]
                p_a = dist_A_pmf[ai]
                # Contribution to squared L² distance
                l2 = l2 + (p_za - p_z * p_a) ** 2

        return l2

    @staticmethod
    def distanceCov(dist, header, Z, A):
        """
        Compute the squared L² distance (distance covariance) between Z and A.

        Parameters:
        - dist (dit.Distribution): Distribution over all variables (features + target + sensitive).
        - header (list[str]): Full list of variable names (order matches dist outcomes).
        - Z (list[str]): Feature(s) to be treated as Z.
        - A (list[str]): Feature(s) to be treated as A.

        Returns:
        - float: Squared L² distance between p(Z, A) and p(Z)p(A).
        """
        if A == [] or Z == []:  # If Z or A is empty, return 0
            return 0
        # Map variable names → indices
        Zi = [header.index(z) for z in Z]
        Ai = [header.index(a) for a in A]
        # Coalesce Z and A into joint distributio
        dist_ZA = dist.coalesce([Zi, Ai])
        dist_ZA.set_rv_names('ZA')  # Set the random variable names for the coalesced distribution


        return DistanceCovariance.distanceCovCore(dist_ZA, 'Z', 'A')

    @staticmethod
    def conditionalDistanceCov(dist, header, Z, A, B):
        """
        Compute the conditional distance covariance dCov(Z, A | B).

        Parameters:
        - dist (dit.Distribution): Distribution over all variables (features + target + sensitive).
        - header (list[str]): Full list of variable names (order matches dist outcomes).
        - Z (list[str]): Feature(s) to be treated as Z.
        - A (list[str]): Feature(s) to be treated as A.
        - B (list[str]): Feature(s) to condition on.

        Returns:
        - float: Conditional distance covariance between Z and A given B.
        """
        if A == [] or Z == []:  # If Z or A is empty, return 0
            return 0
        if B == []:
            return DistanceCovariance.distanceCov(dist, header, Z, A)
        # Map variable names → indices
        Zi = [header.index(z) for z in Z]
        Ai = [header.index(a) for a in A]
        Bi = [header.index(b) for b in B]
        # Coalesce into (Z, A, B) joint distribution
        dist_ZAB = dist.coalesce([Zi, Ai, Bi])
        dist_ZAB.set_rv_names('ZAB')
        # Condition on B and compute weighted average of dCov over conditionals
        marginal, cdists = dist_ZAB.condition_on('B', rvs='ZA')
        marginal_pmf = marginal.pmf
        l = 0.0
        for i in range(len(cdists)):
            l += DistanceCovariance.distanceCovCore(cdists[i], 'Z', 'A') * marginal_pmf[i]
        return l



