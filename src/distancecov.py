

class DistanceCovariance:
    """
    A class to compute distance covariance and distance correlation.
    """

    @staticmethod
    def distanceCovCore(dist, Z, A):

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
                # Compute the squared L2 divergence
                # get two-dimensional index for dist_ZA
                p_za = dist_ZA_pmf[zi * len(dist_A_cordinates) + ai]
                p_z = dist_Z_pmf[zi]
                p_a = dist_A_pmf[ai]
                l2 = l2 + (p_za - p_z * p_a) ** 2

        return l2

    @staticmethod
    def distanceCov(dist, header, Z, A):
        """
        Estimate L² divergence between p(Z, A) and p(Z)p(A) using a dit.Distribution.

        Parameters:
        - dist (dit.Distribution): The distribution over the joint variable space.
        - header (list[str]): Names of all variables in the distribution (order matters).
        - Z (list[str]): Variable(s) to use for Z.
        - A (list[str]): Variable(s) to use for A.

        Returns:
        - float: The squared L² distance between p(Z, A) and p(Z)p(A).
        """
        if A == [] or Z == []:  # If Z or A is empty, return 0
            return 0
        # map feature names to indices
        Zi = [header.index(z) for z in Z]
        Ai = [header.index(a) for a in A]
        dist_ZA = dist.coalesce([Zi, Ai])
        dist_ZA.set_rv_names('ZA')  # Set the random variable names for the coalesced distribution


        return DistanceCovariance.distanceCovCore(dist_ZA, 'Z', 'A')

    @staticmethod
    def conditionalDistanceCov(dist, header, Z, A, B):
        """
        Estimate conditional distance covariance: dCov(Z, A | B).

        Parameters:
        - dist (dit.Distribution): The distribution over the joint variable space.
        - header (list[str]): Names of all variables in the distribution (order matters).
        - Z (list[str]): Variable(s) to use for Z.
        - A (list[str]): Variable(s) to use for A.
        - B (list[str]): Variable(s) to condition on.

        Returns:
        - float: The conditional distance covariance between Z and A given B.
        """
        if A == [] or Z == []:  # If Z or A is empty, return 0
            return 0
        if B == []:
            return DistanceCovariance.distanceCov(dist, header, Z, A)
        # map feature names to indices
        Zi = [header.index(z) for z in Z]
        Ai = [header.index(a) for a in A]
        Bi = [header.index(b) for b in B]
        dist_ZAB = dist.coalesce([Zi, Ai, Bi])  # Coalesce the distribution over Z, A, and B
        dist_ZAB.set_rv_names('ZAB')  # Set the random variable names for the coalesced distribution
        # condition on B
        marginal, cdists = dist_ZAB.condition_on('B', rvs='ZA')
        marginal_pmf = marginal.pmf
        l = 0.0
        for i in range(len(cdists)):
            l += DistanceCovariance.distanceCovCore(cdists[i], 'Z', 'A') * marginal_pmf[i]
        return l



