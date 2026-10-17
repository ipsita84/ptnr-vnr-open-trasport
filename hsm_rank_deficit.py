"""Rank deficit of the collision matrix A for the HSM s=2 band: single node (3x3) and two nodes (6x6).

Same set-up as hsm_rank_check.py: ansatz Lambda_chi = tau_chi(-h_chi + lambda + a cos g + b cos 2g),
overlap T = [3 + 4 cos g cos g' + cos2g cos2g']/8, and the same factor of 1/8 in tau^{-1} and in the
in-scattering term. Numerical rank = number of singular values above 1e-9 * (largest).
Scratch check only; the rank of A does not depend on the driving term h.
"""
import warnings
import numpy as np
from scipy.integrate import quad
warnings.filterwarnings("ignore")

vf, mu = 0.0004, 0.01
phi = [lambda t: 1.0, np.cos, lambda t: np.cos(2 * t)]
wt = np.array([3 / 8, 1 / 2, 1 / 8])
I = lambda g: quad(g, 0, np.pi, epsabs=0, epsrel=1e-12, limit=200)[0]


def weight(chi, B):                      # Fermi-surface weight W_chi(gamma) (notebook-style conventions)
    kf = lambda t: (mu + np.sqrt(mu**2 - 2 * chi * B * np.cos(t)**2)) / (2 * vf)
    D = lambda t: 1 + B * chi * np.cos(t)**2 / kf(t)**2
    vk = lambda t: kf(t) * vf - chi * B * np.cos(t)**2 / (2 * kf(t) * vf)
    return lambda t: np.sin(t) * kf(t)**3 / abs(vk(t)) * D(t)


def matrixA(B, nodes, beta):
    """nodes: tuple of chi values; beta[(chi,chi')]: scattering strengths."""
    W = {c: weight(c, B) for c in nodes}
    tau = {}
    for c in nodes:
        co = [sum(beta[(c, cp)] * wt[j] * I(lambda x, cp=cp, j=j: W[cp](x) * phi[j](x)) for cp in nodes)
              for j in range(3)]
        tau[c] = lambda t, co=co: 1.0 / (co[0] + co[1] * np.cos(t) + co[2] * np.cos(2 * t))
    M = {c: np.array([[I(lambda t: W[c](t) * tau[c](t) * phi[j](t) * phi[k](t)) for k in range(3)]
                      for j in range(3)]) for c in nodes}
    n = len(nodes)
    A = np.zeros((3 * n, 3 * n))
    for a, c in enumerate(nodes):
        for j in range(3):
            A[3 * a + j, 3 * a + j] += 1
            for b, cp in enumerate(nodes):
                A[3 * a + j, 3 * b:3 * b + 3] -= beta[(c, cp)] * wt[j] * M[cp][j]
    return A


def report(label, A):
    sv = np.linalg.svd(A, compute_uv=False)
    rank = int((sv > 1e-9 * sv[0]).sum())
    print(f"{label}: size {A.shape[0]}, rank {rank}, deficit {A.shape[0] - rank}, SV = {np.round(sv, 4)}")


if __name__ == "__main__":
    for B in (0.0, 2e-5, -3e-5):
        report(f"single node, B={B:+.0e}, beta_intra=8", matrixA(B, (1,), {(1, 1): 8.0}))
    for bi in (0.0, 1.0, 4.0, 8.0):
        beta = {(1, 1): 8.0, (-1, -1): 8.0, (1, -1): bi, (-1, 1): bi}
        report(f"two nodes, B=+2e-05, beta_inter={bi}", matrixA(2e-5, (1, -1), beta))
