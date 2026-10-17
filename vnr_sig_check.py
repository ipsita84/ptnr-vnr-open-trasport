"""Python re-implementation of the VNR `sig` function (single node, collinear E and B along z).

Mirrors the Mathematica code: overlap T = (1 + cos g cos g')/2, ansatz Lambda = tau (-h + lam + a cos g),
tau^{-1}(g) = (beta/2) * [ int W0 + cos g * int W0 cos g' ], W0 = kf^2 (kf cos g + k0) D / |vdotk|.

Extras compared with the notebook:
  * kf is written in a form without Sec[g], which is stable at g = pi/2.
  * `null_check` evaluates mat . r, where r = coefficients of 1/tau in the basis (1, cos g).
    It must vanish if the same factor beta/2 enters 1/tau and the in-scattering rows of `mat`.
"""
import warnings
import numpy as np
from scipy.integrate import quad

warnings.filterwarnings("ignore")


def sig(B, mu=0.1, k0=150.0, v0=0.01, beta=0.1, e=1.0):
    kf = lambda g: (mu + (2*k0*v0*mu + mu**2*np.cos(g) - 2*B*e*v0**2*np.sin(g)**2)
                    / (k0*v0 + np.sqrt((k0*v0 + mu*np.cos(g))**2
                                       - 2*B*e*v0**2*np.cos(g)*np.sin(g)**2))) / (2*v0)
    bc = lambda g: -1/(2*kf(g)*(k0 + kf(g)*np.cos(g))) * np.array([np.sin(2*g)/2, 0, np.sin(g)**2])
    D = lambda g: 1 + e*B*bc(g)[2]
    vv = lambda g: v0*np.array([np.cos(g), 0, np.sin(g)])

    def vmm(g):
        K, c, s = kf(g), np.cos(g), np.sin(g)
        return np.array([-(B*e*v0*(2*k0*c + (2 + np.cos(2*g))*K)*s**2)/(2*K*(k0 + c*K)**2),
                         0,
                         (B*e*v0*c**2*s)/(K*(k0 + c*K))])

    vt = lambda g: vv(g) + vmm(g)
    vdk = lambda g: kf(g)*np.dot([np.cos(g), 0, np.sin(g)], vt(g))
    h = lambda g: (vt(g)[2] + e*B*np.dot(bc(g), vt(g)))/D(g)
    W0 = lambda g: kf(g)**2*(kf(g)*np.cos(g) + k0)/abs(vdk(g))*D(g)
    I = lambda f: quad(f, 0, 2*np.pi, epsabs=0, epsrel=1e-11, limit=400)[0]

    z0, c0 = I(W0), I(lambda g: W0(g)*np.cos(g))
    tau = lambda g: 1/(beta/2*z0 + beta*np.cos(g)/2*c0)
    f = lambda g: W0(g)*tau(g)

    unity, cg, cgcg = I(f), I(lambda g: f(g)*np.cos(g)), I(lambda g: f(g)*np.cos(g)**2)
    hunity, hcg = I(lambda g: f(g)*h(g)), I(lambda g: f(g)*h(g)*np.cos(g))

    mat = np.array([[1 - unity*beta/2, -cg*beta/2],
                    [-cg*beta/2,       1 - cgcg*beta/2]])
    colvec = np.array([hunity*beta/2, hcg*beta/2])

    # notebook: row 2 of (mat.x + colvec = 0) together with the number constraint
    M = np.array([[mat[1, 0], mat[1, 1]], [unity, cg]])
    lam, a = np.linalg.solve(M, [-colvec[1], hunity])
    cond = I(lambda g: -f(g)*h(g)*(lam - h(g) + a*np.cos(g)))

    # null-vector check: r = coefficients of 1/tau in (1, cos g)
    r = np.array([beta/2*z0, beta/2*c0])
    sv = np.linalg.svd(mat, compute_uv=False)
    return dict(cond=cond, lam=lam, a=a, hunity=hunity, hcg=hcg, mat=mat, sv=sv,
                null_residual=mat @ r, r=r)


if __name__ == "__main__":
    for B in (0.1, 5.1, 45.1):
        o = sig(B)
        print(f"B={B:5}: cond={o['cond']:.15g}  lam={o['lam']:.1e}  a={o['a']:.1e}")
        print(f"        singular values of mat = {o['sv']}")
        print(f"        |mat.r|/|r| = {np.linalg.norm(o['null_residual'])/np.linalg.norm(o['r']):.1e}")
    print("Notebook values: 0.000999722058657936, 0.0009997219499265218, 0.000999746320010683")
