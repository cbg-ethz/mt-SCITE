import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    from scipy.special import gammaln, logsumexp
    return gammaln, logsumexp, mo, np


@app.cell
def __(gammaln, logsumexp, np):
    # Case 1 - mutation present
    def log_lik_mut(c, vX, vY, vZ, rho):
        """
        Compute likelihood for case 1, when a mutation is present (Eq. 4), i.e.
        P(vY, vZ, vX | M=X; c, rho),
        """

        log_term_1 = p_vYZ_given_vX(c, vX, vY, vZ, rho)
        log_term_2 = p_vX_given_c(c)

        return log_term_1 + log_term_2


    def p_vYZ_given_vX(c, vX, vY, vZ, rho):
        """
        Compute P(vY, vZ | vX, M=X; c, rho), first term from Eq. 4 implemented as
        represented in  Eq. (5).
        Raises ValueError if parameters are outside valid range.
        """
        # Basic sanity checks
        if not (0 <= rho <= 1):
            raise ValueError(f"error rate must be in [0, 1], got {rho}")
        if np.any(np.array([c, vX, vY, vZ]) < 0):
            raise ValueError("Counts must be non-negative.")
        if vX + vY + vZ > c:
            raise ValueError("Sum of counts vX+vY+vZ cannot exceed total c.")

        k = c - vX
        vR = k - vY - vZ
        if vR < 0:
            raise ValueError(f"Invalid reference counts:  r = {vR} < 0")

        log_mult_coeff = (
            gammaln(k + 1)
            - (gammaln(vY + 1) + gammaln(vZ + 1) + gammaln(vR + 1))
        )
        logp = (
            log_mult_coeff
            + (vY + vZ) * np.log(rho / 2.0)
            + vR * np.log(1.0 - rho)
        )

        return logp

    def p_vX_given_c(c):
        """
        Compute P(vX | M=X, c), second term from Eq. 4.
        """
        return (np.log(1) - np.log(c))

    # Case 2 - no mutation present
    def log_lik_nomut(c, vX, vY, vZ, rho):
        """
        Compute likelihood for case 2, when no mutation is present (Eq. 6), i.e.
        P(vY, vZ, vX | M=R; c, rho),
        """

        vR = c - vX - vY - vZ
        vAlt = c - vR

        log_mult_coeff = (
            gammaln(c+1) -
            (gammaln(vX + 1) + gammaln(vY + 1) + gammaln(vZ + 1) + gammaln(vR +1))
        )
        logp = (
            log_mult_coeff + 
            vAlt * np.log(rho / 3.0) +
            vR * np.log(1.0 - rho)
        )

        return logp


    def mutation_probability(c, vX, vY, vZ, rho, prior_mutation_prob):

        # priors
        log_prior_alt = np.log(prior_mutation_prob / 3.0)
        log_prior_ref = np.log(1 - prior_mutation_prob)

        # unnormalised log-posteriors for each mutated state
        Lx = log_lik_mut(c, vX, vY, vZ, rho) + log_prior_alt     # M = X
        Ly = log_lik_mut(c, vY, vX, vZ, rho) + log_prior_alt     # M = Y
        Lz = log_lik_mut(c, vZ, vY, vX, rho) + log_prior_alt     # M = Z
        Lr = log_lik_nomut(c, vX, vY, vZ, rho) + log_prior_ref   # M = R

        # evidence: sum over all four states
        log_evidence = logsumexp([Lx, Ly, Lz, Lr], axis=0)

        # posterior that a mutation exists (any of X, Y, Z)
        log_p_mut = logsumexp([Lx, Ly, Lz], axis=0) - log_evidence

        return np.exp(log_p_mut)
    return (
        log_lik_mut,
        log_lik_nomut,
        mutation_probability,
        p_vX_given_c,
        p_vYZ_given_vX,
    )


@app.cell
def __():
    # example 1 
    # Pos 123, Mut C, Ref A, Mut_ct = 1, Ref_ct = 26
    coverage_pos123 = 27
    vX_pos123 = 1
    vY_pos123 = 0
    vZ_pos123 = 0
    vR_pos123 = 26

    # example 2 
    # Pos 14605, Mut G, Ref A, Mut_ct = 1, Ref_ct = 23
    coverage_pos14605 = 24
    vX_pos14605 = 1
    vY_pos14605 = 0
    vZ_pos14605 = 0
    vR_pos14605 = 23

    error_rate = 0.01
    prior_mut_prob = 10 / 16600
    return (
        coverage_pos123,
        coverage_pos14605,
        error_rate,
        prior_mut_prob,
        vR_pos123,
        vR_pos14605,
        vX_pos123,
        vX_pos14605,
        vY_pos123,
        vY_pos14605,
        vZ_pos123,
        vZ_pos14605,
    )


@app.cell
def __(
    coverage_pos123,
    error_rate,
    mutation_probability,
    prior_mut_prob,
    vX_pos123,
    vY_pos123,
    vZ_pos123,
):
    mutation_probability(c=coverage_pos123, vX=vX_pos123, vY=vY_pos123, vZ=vZ_pos123, rho=error_rate, prior_mutation_prob=prior_mut_prob)
    return


@app.cell
def __(
    coverage_pos14605,
    error_rate,
    mutation_probability,
    prior_mut_prob,
    vX_pos14605,
    vY_pos14605,
    vZ_pos14605,
):
    mutation_probability(c=coverage_pos14605, vX=vX_pos14605, vY=vY_pos14605, vZ=vZ_pos14605, rho=error_rate, prior_mutation_prob=prior_mut_prob)
    return


@app.cell
def __(error_rate, mutation_probability, prior_mut_prob):
    mutation_probability(c=2, vX=2, vY=0, vZ=0, rho=error_rate, prior_mutation_prob=prior_mut_prob)
    return


@app.cell
def __(mo):
    mo.md(r"""Compare to mito functions""")
    return


@app.cell
def __(multinomial, np):
    def _log_prob_ABC_no_mutation(
            reads_R, reads_A, reads_B, reads_C, error_rate):
        """ P(reads_A, reads_B, reads_C | reads_R, no mutation)

        It does not matter which nucleotides are A, B, and C

        Parameters
        ----------
        reads_R : int
            Number of reads for the reference nucleotide
        reads_A, reads_B, reads_C : int
            Number of reads for the non-reference nucleotides

        Returns
        -------
        log_prob_ABC : float
            log P(reads_A, reads_B, reads_C | reads_R, no mutation)
        """
        total_reads = reads_R + reads_A + reads_B + reads_C
        log_prob_ABC = multinomial.logpmf(
           np.dstack([reads_A.tolist(), reads_B.tolist(),
                      reads_C.tolist(), reads_R.tolist()]).squeeze(),
           n=total_reads,
           p=[error_rate / 3, error_rate / 3, error_rate / 3, 1 - error_rate] # p interpretation of rho
        )
        return log_prob_ABC
    return


@app.cell
def __(log_lik_nomut):
    # Test log lik with no mutation
    log_lik_nomut(c=12, vX=1, vY=0, vZ=1, rho=0.01)
    log_lik_nomut(c=0, vX=0, vY=0, vZ=0, rho=0.01)
    log_lik_nomut(c=8, vX=2, vY=1, vZ=0, rho=0.01)
    return


@app.cell
def __(log_lik_mut):
    # Test log lik with  mutation
    log_lik_mut(c=7, vX=7, vY=0, vZ=0, rho=0.05)
    log_lik_mut(c=10, vX=4, vY=1, vZ=0, rho=0.05)
    log_lik_mut(c=0, vX=0, vY=0, vZ=0, rho=0.05)
    return


@app.cell
def __(np):
    np.inf
    return


if __name__ == "__main__":
    app.run()
