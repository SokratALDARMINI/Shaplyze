"""
information.py

Information-theoretic measures for Shaplyze.

Functions:
    mi(dist, header, Z, A): Mutual information I(Z; A).
    cmi(dist, header, Z, A, B): Conditional mutual information I(Z; A | B).
    entropy(dist, header, Z): Entropy H(Z).
    cond_entropy(dist, header, Z, A): Conditional entropy H(Z|A).
    nmi(dist, header, Z, A): Normalized mutual information.
    uni(dist, header, Z, A, B): Unique information Uni(Z : A | B).
    red(dist, header, Z, A, B): Redundant information Red(Z:(A,B)).
    syn(dist, header, Z, A, B): Synergistic information Syn(Z:A|B).
"""
from dit.shannon import mutual_information as _I, entropy as _H

from src import BROJA_2PID
from src import BROJA_2PID_MOSEK

class MIDist:
    """
    Collection of mutual-information–based measures using feature lists
    and `dit` distributions, with BROJA-based PID for unique, redundant,
    and synergistic information.
    """
    @staticmethod
    def mi(dist, header, Z, A):
        """
        Compute mutual information I(Z; A).

        Parameters:
            dist : dit.Distribution
                Joint distribution over variables.
            header : list[str]
                List of variable names in the order used to build `dist`.
            Z : list[str]
                Names of the variable subset Z.
            A : list[str]
                Names of the variable subset A.

        Returns:
            float
                Mutual information in bits.
        """
        if A == [] or Z == []:  # If Z or A is empty, return 0
            return 0
        # map feature names to indices
        Zi = [header.index(z) for z in Z]
        Ai = [header.index(a) for a in A]
        dm = dist.coalesce([Zi, Ai])
        return _I(dm, [0], [1])

    @staticmethod
    def cmi(dist, header, Z, A, B):
        """
        Compute conditional mutual information I(Z; A | B).

        Parameters:
            dist : dit.Distribution
            header : list[str]
            Z : list[str]
            A : list[str]
            B : list[str]

        Returns:
            float
                Conditional mutual information in bits.
        """
        return MIDist.mi(dist, header, Z, A + B) - MIDist.mi(dist, header, Z, B)

    @staticmethod
    def entropy(dist, header, Z):
        """
        Compute entropy H(Z).

        Parameters:
            dist : dit.Distribution
            header : list[str]
            Z : list[str]

        Returns:
            float
                Entropy in bits.
        """
        if Z == []:  # If Z is empty, return 0
            return 0
        Zi = [header.index(z) for z in Z]
        dm = dist.coalesce([Zi])
        return _H(dm)

    @staticmethod
    def cond_entropy(dist, header, Z, A):
        """
        Compute conditional entropy H(Z | A).

        H(Z|A) = H(Z,A) - H(A)

        Parameters:
            dist : dit.Distribution
            header : list[str]
            Z : list[str]
            A : list[str]

        Returns:
            float
                Conditional entropy in bits.
        """
        if Z == []:
            return 0.0
        if A == []:
            return MIDist.entropy(dist, header, Z)
        Zi = [header.index(z) for z in Z]
        Ai = [header.index(a) for a in A]
        hZA = _H(dist.coalesce([Zi + Ai]))
        hA = _H(dist.coalesce([Ai]))
        return hZA - hA

    @staticmethod
    def nmi(dist, header, Z, A):
        """
        Compute normalized mutual information:
            NMI = 2 * I(Z; A) / (H(Z) + H(A)).

        Parameters:
            dist : dit.Distribution
            header : list[str]
            Z : list[str]
            A : list[str]

        Returns:
            float
                Normalized mutual information between 0 and 1.
        """
        i_val = MIDist.mi(dist, header, Z, A)
        hZ = MIDist.entropy(dist, header, Z)
        hA = MIDist.entropy(dist, header, A)
        denom = hZ + hA
        return (2 * i_val / denom) if denom > 0 else 0.0

    @staticmethod
    def uni(dist, header, Z, A, B, eps=1e-7):
        """
           Compute the unique information component Uni(Z : A | B) using the BROJA partial information decomposition.

           Parameters
           ----------
           dist : dit.base.Dist
               A multivariate discrete distribution object from the 'dit' library, representing variables S, X, Y, etc.
           header : list of str
               Ordered list of random variable names in 'dist'. The index of each variable in 'header' must correspond
               to its coordinate in the distribution support.
           Z : list of str
               Names of the target (or output) variables whose unique information from A (conditioned on B) is computed.
           A : list of str
               Names of the source variable(s) for which we measure unique information about Z, relative to B.
           B : list of str
               Names of the second source (background) variables that may also share information about Z.

           Returns
           -------
           float
               The unique information UIY = Uni(Z : A | B) in bits, as returned by the BROJA PID solver.

           Raises
           ------
           Exception
               If both ECOS and MOSEK solvers fail to produce a valid decomposition within the specified tolerance,
               an exception is raised indicating failure for subset Z, A, B.

           Notes
           -----
           - Internally, the function:
             1. Extracts indices of Z, A, B in 'header'.
             2. Coalesces the distribution `dist` onto these variables, making it dense.
             3. Constructs the joint PMF `pdf[(i,j,k)]` over (S, X, Y) tuples.
             4. Normalizes the PMF to ensure it sums to 1.
             5. Computes mutual information values m1 = I(Z;A), cm1 = I(Z;A|B), m2 = I(Z;B), cm2 = I(Z;B|A) via `mi` and `cmi`.
             6. Attempts PID computation with MOSEK first (for large support or multi-dimensional Z), then falls back to ECOS.
             7. Validates solver output against error tolerances `eps`, rejecting any solution that violates bounds.
             8. Returns the `UIY` component (unique information for A) or raises if both solvers fail.
           """

        if not Z or not A:
            return 0.0
        if not B:
            return MIDist.mi(dist, header, Z, A)
        T = Z
        Z1 = A
        Z2 = B
        dis = dist
        Ti = [header.index(t) for t in T]
        Z1i = [header.index(z1) for z1 in Z1]
        Z2i = [header.index(z2) for z2 in Z2]

        P = dis.coalesce([Ti, Z1i, Z2i])
        P.make_dense()
        # collect the marginals
        P.set_rv_names('SXY')
        Ps = P.coalesce('S')
        Ps_cordinates = [i for i in range(len(Ps.outcomes))]

        Pz1 = P.coalesce('X')
        pz1_cordinates = [i for i in range(len(Pz1.outcomes))]

        Pz2 = P.coalesce('Y')
        pz2_cordinates = [i for i in range(len(Pz2.outcomes))]

        pmf = P.pmf.reshape(len(Ps_cordinates), len(pz1_cordinates), len(pz2_cordinates))

        pdf = dict()
        for i in Ps_cordinates:
            for j in pz1_cordinates:
                for k in pz2_cordinates:
                    pdf[(i, j, k)] = float(pmf[i, j, k])

        m = sum(pdf.values())
        # Normalize PDF
        for key in pdf:
            pdf[key] = pdf[key] / m

        parms = dict()
        parms['max_iters'] = 3000

        # Preliminary MI and CMI diagnostics
        m1 = MIDist.mi(dist, header, Z, A)
        cm1 = MIDist.cmi(dist, header, Z, A, B)
        m2 = MIDist.mi(dist, header, Z, B)
        cm2 = MIDist.cmi(dist, header, Z, B, A)

        faultMOSEK = False
        faultECOS = False

        Z_large_support = False

        if len(Z) > 1 or len(Ps.outcomes) > 2:
            Z_large_support = True

        # Try ECOS first on large support
        if Z_large_support:
            returndataECOS = BROJA_2PID.pid(pdf, cone_solver="ECOS", output=2, **parms)
            if max(list(returndataECOS['Num_err'])) > 0.001:
                faultECOS = True

            if not faultECOS:
                return returndataECOS['UIY']

            # Fall back to MOSEK
            returndataMOSEK = BROJA_2PID_MOSEK.pid(pdf, cone_solver="MOSEK", output=2, **parms)
            if max(list(returndataMOSEK['Num_err'])) > 0.01 or (returndataMOSEK['UIY'] < -eps) or (
                    returndataMOSEK['UIY'] > m1 + eps) or (returndataMOSEK['UIY'] > cm1 + eps) or (
                    returndataMOSEK['UIZ'] < -eps) or (returndataMOSEK['UIZ'] > m2 + eps) or (
                    returndataMOSEK['UIZ'] > cm2 + eps) or (returndataMOSEK['SI'] < -eps) or (
                    returndataMOSEK['CI'] < -eps):
                faultMOSEK = True

            if not faultMOSEK:
                print("ECOS failed for subset Z=", Z, "A=", A, "B=", B,
                      "Warning, MOSEK calculated without errors. Note that MOSEK is not reliable for large support")
                print("returndataMOSEK=", returndataMOSEK)
                print("returndataECOS=", returndataECOS)
                return returndataMOSEK['UIY']

            print("*************************************************************************************************")
            print("Both MOSEK and ECOS failed for subset Z=", Z, "A=", A, "B=", B)
            print("returndataMOSEK=", returndataMOSEK)
            print("returndataECOS=", returndataECOS)
            print("*************************************************************************************************")
            raise Exception("Both MOSEK and ECOS failed for subset Z=", Z, "A=", A, "B=", B)
            l = [m1, cm1]
            return min(l)

        # For small support, try MOSEK first
        returndataMOSEK = BROJA_2PID_MOSEK.pid(pdf, cone_solver="MOSEK", output=2, **parms)
        if max(list(returndataMOSEK['Num_err'])) > 0.01 or (returndataMOSEK['UIY'] < -eps) or (
                returndataMOSEK['UIY'] > m1 + eps) or (returndataMOSEK['UIY'] > cm1 + eps) or (
                returndataMOSEK['UIZ'] < -eps) or (returndataMOSEK['UIZ'] > m2 + eps) or (
                returndataMOSEK['UIZ'] > cm2 + eps) or (returndataMOSEK['SI'] < -eps) or (
                returndataMOSEK['CI'] < -eps):
            faultMOSEK = True

        if not faultMOSEK:
            return returndataMOSEK['UIY']

        returndataECOS = BROJA_2PID.pid(pdf, cone_solver="ECOS", output=2, **parms)
        if max(list(returndataECOS['Num_err'])) > 0.001:
            faultECOS = True

        if not faultECOS:
            print("MOSEK failed for subset Z=", Z, "A=", A, "B=", B, "But ECOS calculated without errors")
            print("returndataMOSEK=", returndataMOSEK)
            print("returndataECOS=", returndataECOS)
            return returndataECOS['UIY']

        print("*************************************************************************************************")
        print("Both MOSEK and ECOS failed for subset Z=", Z, "A=", A, "B=", B)
        print("returndataMOSEK=", returndataMOSEK)
        print("returndataECOS=", returndataECOS)
        print("*************************************************************************************************")
        raise Exception("Both MOSEK and ECOS failed for subset Z=", Z, "A=", A, "B=", B)
        l = [m1, cm1]
        return min(l)

    @staticmethod
    def red(dist, header, Z, A, B):
        """
        Redundant information Red(Z:(A,B)) for discrete variables via partial information decomposition:
            Red(Z:(A,B)) = I(Z;A) - Uni(Z:A|B)

        Parameters
        ----------
        dist
            A multivariate discrete distribution (from dit).
        header : list[str]
            Variable ordering corresponding to dist.
        Z : list[str]
            Target variable(s).
        A : list[str]
            First source variable(s).
        B : list[str]
            Second source variable(s).

        Returns
        -------
        float
            The redundant information component in bits.
        """
        return MIDist.mi(dist, header, Z, A) - MIDist.uni(dist, header, Z, A, B)

    @staticmethod
    def syn(dist, header, Z, A, B):
        """
        Synergistic information Syn(Z:A|B) for discrete variables:
            Syn(Z:A|B) = I(Z;A|B) - Uni(Z:A|B)

        Parameters
        ----------
        dist :
            A multivariate discrete distribution (from dit).
        header : list[str]
            Variable ordering corresponding to dist.
        Z : list[str]
            Target variable(s).
        A : list[str]
            First source variable(s).
        B : list[str]
            Second source variable(s).

        Returns
        -------
        float
            The synergistic information component in bits.
        """
        return MIDist.cmi(dist, header, Z, A, B) - MIDist.uni(dist, header, Z, A, B)