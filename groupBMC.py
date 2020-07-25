""" Bayesian model selection for group studies.
Adapted from VBA-toolbox (https://github.com/MBB-team/VBA-toolbox) by Lionel Rigoux.
References:
[1] Rigoux, L., Stephan, K. E., Friston, K. J., & Daunizeau, J. (2014).
Bayesian model selection for group studies—revisited. NeuroImage, 84, 971-985.
https://www.tnu.ethz.ch/fileadmin/user_upload/documents/Publications/2014/2014_Rigoux_Stephan_Friston_Daunizeau.pdf.
[2] Stephan, K. E., Penny, W. D., Daunizeau, J., Moran, R. J., & Friston, K. J. (2009).
Bayesian model selection for group studies. NeuroImage, 46(4), 1004-1017.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2703732/pdf/ukmss-5226.pdf.
"""
__author__ = 'Sichao Yang'
__contact__ = 'sichao@cs.wisc.edu'
__license__ = 'MIT'


from typing import List, Optional
from math import exp, log
import numpy as np
from scipy import integrate
from scipy.stats import rv_continuous, dirichlet, multivariate_normal as mvn
from scipy.special import digamma as ψ, gammainc, gammaln, softmax, expit
ε: float = np.finfo(float).eps


def exceedance_probability(distribution: rv_continuous, n_samples: Optional[int] = None):
    """ Calculates the exceedance probability of a random variable following a continuous multivariate distribution.
    Exceedance probability: φ_i = p(∀j != i: x_i > x_j | x ~ ``distribution``).

    :param distribution: the continuous multivariate distribution.
    :param n_samples: the number of realization sampled from the distribution to approximate the exceedance probability.
                      Default to ``None`` and numerical integration is used instead of Monte Carlo simulation.
    :return: the exceedance probability of a random variable following the continuous multivariate distribution.
    """
    if n_samples is None:   # Numerical integration
        from scipy.stats._multivariate import dirichlet_frozen, multivariate_normal_frozen
        if type(distribution) is multivariate_normal_frozen:
            # Speekenbrink, M., & Konstantinidis, E. (2015). Uncertainty and exploration in a restless bandit problem.
            # https://onlinelibrary.wiley.com/doi/pdf/10.1111/tops.12145: p. 4.
            distribution: multivariate_normal_frozen
            μ, Σ = distribution.mean, distribution.cov
            n = len(μ)
            φ = np.zeros(n)
            I = - np.eye(n - 1)
            for i in range(n):
                A = np.insert(I, i, 1, axis=1)
                φ[i] = (mvn.cdf(A @ μ, cov=A @ Σ @ A.T))
        elif type(distribution) is dirichlet_frozen:
            # Soch, J. & Allefeld, C. (2016). Exceedance Probabilities for the Dirichlet Distribution.
            # https://arxiv.org/pdf/1611.01439.pdf: p. 361.
            distribution: dirichlet_frozen
            α = distribution.alpha
            n = len(α)
            γ = [gammaln(α[i]) for i in range(n)]

            def f(x, i):
                φ_i = 1
                for j in range(n):
                    if i != j:
                        φ_i *= gammainc(α[j], x)
                return φ_i * exp((α[i] - 1) * log(x) - x - γ[i])
            φ = [integrate.quad(lambda x: f(x, i), 0, np.inf)[0] for i in range(n)]
        else:
            raise NotImplementedError('Numerical integration not implemented for this distribution!')
        φ = np.array(φ)
    else:   # Monte Carlo simulation
        samples = distribution.rvs(size=n_samples)
        φ = (samples == np.amax(samples, axis=1, keepdims=True)).sum(axis=0)
    return φ / φ.sum()


class GroupBMCResult:
    """ Results of Bayesian model selection for group studies. """
    attribution: np.ndarray     # posterior probabilities for each subject to belong to each model/family
    frequency_mean: np.ndarray  # mean of the posterior Dirichlet distribution on model/family frequencies
    frequency_var: np.ndarray   # variance of the posterior Dirichlet distribution on model/family frequencies
    exceedance_probability: np.ndarray              # p. 972
    protected_exceedance_probability: np.ndarray    # p. 973 (7)

    def __init__(self, α: np.ndarray, z: np.ndarray, bor: float):
        """
        :param α: sufficient statistics of the posterior Dirichlet density on model/family frequencies
        :param z: posterior probabilities for each subject to belong to each model/family
        :param bor: Bayesian omnibus risk p(y|H0)/(p(y|H0)+p(y|H1))
        """
        self.attribution = z.copy()
        self.frequency_mean = dirichlet.mean(α)
        self.frequency_var = dirichlet.var(α)
        self.exceedance_probability = exceedance_probability(dirichlet(α))
        self.protected_exceedance_probability = self.exceedance_probability * (1 - bor) + bor / len(α)  # (7)


class GroupBMC:
    """ Variational Bayesian algorithm for group-level Bayesian Model Comparison.
    Rigoux, L., Stephan, K. E., Friston, K. J., & Daunizeau, J. (2014).
    Bayesian model selection for group studies—revisited.
    https://www.tnu.ethz.ch/fileadmin/user_upload/documents/Publications/2014/2014_Rigoux_Stephan_Friston_Daunizeau.pdf.
    """
    L: np.ndarray           # KxN array of the log-evidence of each model given each subject
    families: np.ndarray    # KxNf array of the attribution of each model to each family
    α_0: np.ndarray         # Kx1 array of sufficient statistics of the prior Dirichlet density on model frequencies
    α: np.ndarray           # Kx1 array of sufficient statistics of the posterior Dirichlet density on model frequencies
    z: np.ndarray           # KxN array of posterior probabilities for each subject to belong to each model
    F: List[float]          # The series of free energies along the VB iterations

    def __init__(self,
                 L:             np.ndarray,
                 α_0:           Optional[np.ndarray] = None,
                 partitions:    Optional[List[List[int]]] = None,
                 max_iter:      int = 32,
                 min_iter:      int = 1,
                 tolerance:     float = 1e-4):
        """ Uses variational Bayesian analysis to fit a Dirichlet distribution on model frequencies to the data.

        :param L: KxN array of the log-evidence of each of the K models given each of the N subjects.
        :param α_0: Kx1 array of sufficient statistics of the prior Dirichlet density of model frequencies.
        :param partitions: Nfx1 array of arrays pf indices (1 to K) of models belonging to each of the Nf families.
        :param max_iter: max number of iterations.
        :param min_iter: min number of iterations.
        :param tolerance: max change in free energy.
        """
        self.L = L
        K, N = L.shape
        partitions = [np.array([i]) for i in range(K)] if partitions is None else [np.array(p) - 1 for p in partitions]
        assert np.all(np.sort(np.concatenate(partitions)) == np.arange(K)), 'Invalid partition of the model space!'
        Nf = len(partitions)
        self.families = np.zeros((K, Nf), dtype=np.bool)
        for j in range(Nf):
            self.families[partitions[j], j] = True
        self.α_0 = (self.families / self.families.sum(axis=0) @ (np.ones(Nf) / Nf) if α_0 is None else α_0)[:, None]
        assert len(self.α_0) == K, 'Model evidence and priors size mismatch!'
        self.α, self.z = self.α_0.copy(), np.tile(self.α_0, (1, N))

        self.F = []
        for i in range(1, max_iter + 1):
            self.z = softmax(self.L + ψ(self.α), axis=0)            # (A21) line 2
            self.α = self.α_0 + self.z.sum(axis=1, keepdims=True)   # (A21) line 1

            self.F.append(self.F1())
            if i > max(min_iter, 1) and abs(self.F[-1] - self.F[-2]) < tolerance:
                break
    
    def get_result(self) -> GroupBMCResult:
        """ Get various statistics of the posterior Dirichlet distribution on model frequencies. """
        bor: float = 1 / (1 + exp(self.F1() - self.F0()))
        if self.families.size == 0:
            return GroupBMCResult(self.α.flatten(), self.z, bor)
        return GroupBMCResult(self.families.T @ self.α.flatten(), self.families.T @ self.z, bor)

    def F0(self) -> float:
        """ Derives the free energy of the null hypothesis (H0: uniform priors). """
        w = softmax(self.L, axis=0)                                      # (A19)
        return (w * (self.L + np.log(self.α_0) - np.log(w + ε))).sum()   # (A17)

    def F1(self) -> float:
        """ Derives the free energy for the current approximate posteriors (H1). """
        E_log_r = (ψ(self.α) - ψ(self.α.sum()))
        E_log_joint = (self.z * (self.L + E_log_r)).sum() + ((self.α_0 - 1) * E_log_r).sum()        # (A20) line 2
        E_log_joint += gammaln(self.α_0.sum()) - gammaln(self.α_0).sum()                            # (A20) line 3
        entropy_z = -(self.z * np.log(self.z + ε)).sum()                                            # (A20) line 3
        entropy_α = gammaln(self.α).sum() - gammaln(self.α.sum()) - ((self.α - 1) * E_log_r).sum()  # (A20) line 4
        return E_log_joint + entropy_z + entropy_α
