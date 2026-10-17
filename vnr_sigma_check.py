"""Python reimplementation of the VNR `sig` function (single node, s=2 band, B along z).

Mirrors the Mathematica notebook scatt-mat-vnr_bz.nb:
  overlap T = (1 + cos g cos g')/2, ansatz Lambda/tau = lam - h + a cos g, 2x2 matrix mat,
  Solve[{row 2, number constraint}], cond = -Int f h Lambda.
Differences: kf is written in a numerically stable form (no Sec[g] cancellation), phi = 0.
Also implements the null-vector check  mat . r = 0, with r = harmonic coefficients of 1/tau.
"""
import warnings
import numpy as np
from scipy.integrate import quad
warnings.filterwarnings("ignore")


def run(B, mu=0.1, k0=4.0, v0=1.0, beta=0.1, e=1.0):
    c, s = np.cos, np.sin

    def kf(g):
        X = (k0 * v0 + mu * c(g)) ** 2 - 2 * B * e * v0**2 * c(g) * s(g) ** 2
        return (mu + (2 * k0 * v0 * mu + mu**2 * c(g) - 2 * B * e * v0**2 * s(g) ** 2)
                / (k0 * v0 + np.sqrt(X))) / (2 * v0)

    def bc(g):
        return -1 / (2 * kf(g) * (k0 + kf(g) * c(g))) * np.array([s(2 * g) / 2, 0, s(g) ** 2])

    D = lambda g: 1 + e * B * bc(g)[2]

    def vtot(g):
        K = kf(g)
        vx = v0 * c(g) - (B * e * v0 * (2 * k0 * c(g) + (2 + c(2 * g)) * K) * s(g) ** 2) / (2 * K * (k0 + c(g) * K) ** 2)
        vz = v0 * s(g) + (B * e * v0 * c(g) ** 2 * s(g)) / (K * (k0 + c(g) * K))
        return np.array([vx, 0, vz])

    vdotk = lambda g: kf(g) * np.dot([c(g), 0, s(g)], vtot(g))
    h = lambda g: (vtot(g)[2] + e * B * np.dot(bc(g), vtot(g))) / D(g)
    W0 = lambda g: kf(g) ** 2 * (kf(g) * c(g) + k0) / abs(vdotk(g)) * D(g)   # weight without tau
    I = lambda f: quad(f, 0, 2 * np.pi, epsabs=0, epsrel=1e-11, limit=400)[0]

    # harmonic coefficients of 1/tau:  1/tau = r0 + r1 cos g   (t_00 = t_11 = 1/2)
    r = np.array([beta / 2 * I(W0), beta / 2 * I(lambda g: W0(g) * c(g))])
    tau = lambda g: 1 / (r[0] + r[1] * c(g))
    f = lambda g: W0(g) * tau(g)

    zero, cg, cgcg = I(f), I(lambda g: f(g) * c(g)), I(lambda g: f(g) * c(g) ** 2)
    hzero, hcg = I(lambda g: f(g) * h(g)), I(lambda g: f(g) * h(g) * c(g))
    mat = np.array([[1 - zero * beta / 2, -cg * beta / 2],
                    [-cg * beta / 2, 1 - cgcg * beta / 2]])
    colvec = np.array([hzero * beta / 2, hcg * beta / 2])

    # notebook: Solve[{matr[[2]] + colvec[[2]] == 0, cs == 0}, {lam, a}]
    lam, a = np.linalg.solve(np.array([[mat[1, 0], mat[1, 1]], [zero, cg]]),
                             np.array([-colvec[1], hzero]))
    Lam = lambda g: lam - h(g) + a * c(g)
    cond = I(lambda g: -f(g) * h(g) * Lam(g))

    sv = np.linalg.svd(mat, compute_uv=False)
    resid = np.abs(mat @ r).max() / np.abs(r).max()        # the mat . r = 0 check
    return dict(cond=cond, lam=lam, a=a, hzero=hzero, hcg=hcg, sv=sv, resid=resid)


if __name__ == "__main__":
    print("test table of the notebook (mu=0.1, k0=150, v0=0.01, beta=0.1):")
    for B in (0.1, 5.1, 45.1):
        o = run(B, mu=0.1, k0=150.0, v0=0.01, beta=0.1)
        print(f"B={B:5.1f}  cond={o['cond']:.15e}  lam,a={o['lam']:.1e},{o['a']:.1e}  "
              f"hzero,hcg={o['hzero']:.1e},{o['hcg']:.1e}  sv={o['sv']}  |mat.r|/|r|={o['resid']:.1e}")
