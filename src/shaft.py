"""Gear shaft sizing with a finite element beam model and a material comparison.

SELF INITIATED STUDY. Every load, geometry and material value below is ASSUMED, see README.
Run:  python src/shaft.py   (writes results/)
"""
import csv, json, math, pathlib
import numpy as np
from scipy.linalg import eigh

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"

# ------------------------------------------------------------------ ASSUMED INPUTS
P_W, N_RPM = 12e3, 6000.0                  # transmitted power, shaft speed
D_PITCH, ALPHA_N, BETA = 0.060, math.radians(20), math.radians(20)   # gear pitch dia, normal pressure angle, helix angle
L = 0.160                                  # bearing span, m
X_GEAR = 0.060                             # gear position from bearing A, m
SEG = [(0.000, 0.030, 0.020), (0.030, 0.130, 0.025), (0.130, 0.160, 0.020)]  # (x0, x1, diameter) m
M_GEAR, J_GEAR = 0.8, 4.0e-4               # gear mass kg and polar mass inertia kg m^2 (lumped, not used for stiffness)
KF = 1.5                                   # fatigue stress concentration, whole shaft, conservative
LIM = dict(sf_static=2.0, sf_fatigue=1.5, f1_ratio=1.5)   # acceptance limits, assumed
# material: E Pa, rho kg/m3, Sy Pa (or strength limit), Sut Pa, endurance ratio Se'/Sut, surface factor, relative price per kg (steel = 1)
MATS = {
    "AISI 4140 steel":      dict(E=205e9, rho=7850, Sy=650e6, Sut=900e6, er=0.50, ka=0.80, price=1.0),
    "Ductile cast iron":    dict(E=169e9, rho=7100, Sy=370e6, Sut=600e6, er=0.40, ka=0.75, price=0.9),
    "Cast aluminium A356":  dict(E=72e9,  rho=2680, Sy=185e6, Sut=255e6, er=0.35, ka=0.75, price=2.5),
    "PA66 GF30 polymer":    dict(E=9e9,   rho=1370, Sy=100e6, Sut=180e6, er=0.25, ka=0.90, price=3.0),
}

def diameter_at(x, segs=SEG):
    for x0, x1, d in segs:
        if x0 - 1e-12 <= x <= x1 + 1e-12:
            return d
    raise ValueError(x)

# ------------------------------------------------------------------ statics
def loads():
    T = P_W / (2 * math.pi * N_RPM / 60)
    Ft = 2 * T / D_PITCH
    Fr = Ft * math.tan(ALPHA_N) / math.cos(BETA)
    Fa = Ft * math.tan(BETA)
    return T, Ft, Fr, Fa

def bending_moment(x, F, a, Mc=0.0, span=L):
    """Simply supported beam, point load F at a, applied couple Mc at a. Closed form."""
    b = span - a
    RA = F * b / span - Mc / span
    M = RA * x - np.where(x > a, F * (x - a), 0.0) + np.where(x > a, Mc, 0.0)
    return M

# ------------------------------------------------------------------ beam finite elements
def beam_matrices(EI, rhoA, le):
    k = EI / le**3 * np.array([[12, 6*le, -12, 6*le], [6*le, 4*le**2, -6*le, 2*le**2],
                               [-12, -6*le, 12, -6*le], [6*le, 2*le**2, -6*le, 4*le**2]])
    m = rhoA * le / 420 * np.array([[156, 22*le, 54, -13*le], [22*le, 4*le**2, 13*le, -3*le**2],
                                    [54, 13*le, 156, -22*le], [-13*le, -3*le**2, -22*le, 4*le**2]])
    return k, m

def build_beam(E, rho, nel_per_mm=0.5, span=L, segs=SEG, x_gear=X_GEAR, uniform_d=None):
    """Nodes are placed so that segment ends and the gear position are nodes."""
    xs = {0.0, span, x_gear}
    for x0, x1, _ in segs:
        xs.update([x0, x1])
    xs = sorted(xs)
    nodes = [xs[0]]
    for a, b in zip(xs[:-1], xs[1:]):
        n = max(1, int(round((b - a) * 1000 * nel_per_mm)))
        nodes += list(np.linspace(a, b, n + 1)[1:])
    nodes = np.array(nodes)
    nn = len(nodes)
    K = np.zeros((2*nn, 2*nn)); M = np.zeros_like(K)
    for e in range(nn - 1):
        le = nodes[e+1] - nodes[e]
        d = uniform_d if uniform_d else diameter_at(0.5 * (nodes[e] + nodes[e+1]), segs)
        I, A = math.pi * d**4 / 64, math.pi * d**2 / 4
        k, m = beam_matrices(E * I, rho * A, le)
        s = slice(2*e, 2*e + 4)
        K[s, s] += k; M[s, s] += m
    return nodes, K, M

def solve_static(nodes, K, F, a, Mc=0.0):
    nn = len(nodes); f = np.zeros(2*nn)
    ig = int(np.argmin(abs(nodes - a)))
    f[2*ig] += F; f[2*ig + 1] += Mc
    fixed = [0, 2*(nn - 1)]                      # w = 0 at both bearings, rotations free
    free = [i for i in range(2*nn) if i not in fixed]
    u = np.zeros(2*nn); u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    return u, free

def modal(nodes, K, M, n_modes=3, gear=True, x_gear=X_GEAR):
    nn = len(nodes); Mm = M.copy()
    if gear:
        ig = int(np.argmin(abs(nodes - x_gear)))
        Mm[2*ig, 2*ig] += M_GEAR
        Mm[2*ig + 1, 2*ig + 1] += J_GEAR * 0.5   # diametral inertia taken as half the polar value
    fixed = [0, 2*(nn - 1)]
    free = [i for i in range(2*nn) if i not in fixed]
    w2 = eigh(K[np.ix_(free, free)], Mm[np.ix_(free, free)], eigvals_only=True)[:n_modes]
    return np.sqrt(w2) / (2 * math.pi)

# ------------------------------------------------------------------ validation against closed form
def validation():
    E, rho, d = MATS["AISI 4140 steel"]["E"], MATS["AISI 4140 steel"]["rho"], 0.025
    I, A = math.pi * d**4 / 64, math.pi * d**2 / 4
    rows = []
    for npm in (0.1, 0.25, 0.5, 1.0):
        nodes, K, M = build_beam(E, rho, nel_per_mm=npm, segs=[(0.0, L, d)], x_gear=L/2, uniform_d=d)
        u, _ = solve_static(nodes, K, 1000.0, L/2)
        w_fe = u[2*int(np.argmin(abs(nodes - L/2)))]
        w_cf = 1000.0 * L**3 / (48 * E * I)
        f_fe = modal(nodes, K, M, 3, gear=False)
        f_cf = [(n*math.pi)**2 / (2*math.pi*L**2) * math.sqrt(E*I/(rho*A)) for n in (1, 2, 3)]
        rows.append(dict(elements=len(nodes)-1, defl_fe_mm=w_fe*1e3, defl_cf_mm=w_cf*1e3,
                         defl_err_pct=100*abs(w_fe-w_cf)/w_cf,
                         f1_err_pct=100*abs(f_fe[0]-f_cf[0])/f_cf[0], f2_err_pct=100*abs(f_fe[1]-f_cf[1])/f_cf[1],
                         f3_err_pct=100*abs(f_fe[2]-f_cf[2])/f_cf[2], f1_cf_hz=f_cf[0]))
    return rows

# ------------------------------------------------------------------ material study
def study(name, mat, scale=1.0):
    T, Ft, Fr, Fa = loads()
    segs = [(a, b, d*scale) for a, b, d in SEG]
    nodes, K, M = build_beam(mat["E"], mat["rho"], nel_per_mm=0.5, segs=segs)
    u1, free = solve_static(nodes, K, Ft, X_GEAR)                      # tangential plane
    u2, _ = solve_static(nodes, K, Fr, X_GEAR, Mc=Fa*D_PITCH/2)        # radial plane with axial couple
    w = np.hypot(u1[0::2], u2[0::2])
    defl_max = float(w.max())
    x = nodes
    Mres = np.hypot(bending_moment(x, Ft, X_GEAR), bending_moment(x, Fr, X_GEAR, Mc=Fa*D_PITCH/2))
    d = np.array([diameter_at(min(max(xx, 1e-9), L-1e-9), segs) for xx in x])
    sig_b = 32*Mres/(math.pi*d**3); tau = 16*T/(math.pi*d**3)
    # stations at the very ends carry zero moment, evaluate everywhere
    s_vm = np.sqrt(sig_b**2 + 3*tau**2)
    sf_static = mat["Sy"] / s_vm.max()
    Se = mat["er"] * mat["Sut"] * mat["ka"]
    sa = KF * sig_b; sm = np.sqrt(3) * KF * tau
    sf_fat = float(np.min(1.0 / (sa/Se + sm/mat["Sut"] + 1e-30)))
    f = modal(nodes, K, M, 3)
    mass = sum(mat["rho"] * math.pi * diameter_at(0.5*(nodes[e]+nodes[e+1]), segs)**2/4 * (nodes[e+1]-nodes[e]) for e in range(len(nodes)-1))
    f_op = N_RPM / 60
    row = dict(material=name, diameter_scale=scale, mass_kg=mass, price_index=mass*mat["price"], defl_max_um=defl_max*1e6,
               sf_static=float(sf_static), sf_fatigue=sf_fat, f1_hz=float(f[0]), f2_hz=float(f[1]), f3_hz=float(f[2]),
               f1_over_speed=float(f[0]/f_op))
    row["pass"] = bool(row["sf_static"] >= LIM["sf_static"] and row["sf_fatigue"] >= LIM["sf_fatigue"] and row["f1_over_speed"] >= LIM["f1_ratio"])
    return row

def min_size(name, mat):
    """Smallest uniform diameter scale that meets all assumed limits, found by bisection."""
    lo, hi = 0.2, 4.0
    ok = lambda r: r["pass"]
    if not ok(study(name, mat, hi)):
        return None
    for _ in range(40):
        mid = 0.5*(lo+hi)
        if ok(study(name, mat, mid)): hi = mid
        else: lo = mid
    r = study(name, mat, hi)
    gov = min((("static strength", r["sf_static"]/LIM["sf_static"]), ("fatigue strength", r["sf_fatigue"]/LIM["sf_fatigue"]),
               ("first natural frequency", r["f1_over_speed"]/LIM["f1_ratio"])), key=lambda t: t[1])[0]
    r["governing"] = gov
    return r

def main():
    OUT.mkdir(exist_ok=True)
    T, Ft, Fr, Fa = loads()
    val = validation()
    rows = [study(n, m) for n, m in MATS.items()]
    sized = [min_size(n, m) for n, m in MATS.items()]
    for path, data in (("validation.csv", val), ("materials.csv", rows), ("min_size.csv", sized)):
        with open(OUT / path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    summary = dict(torque_Nm=T, Ft_N=Ft, Fr_N=Fr, Fa_N=Fa, validation=val, materials=rows, min_size=sized, limits=LIM)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].loglog([r["elements"] for r in val], [r["f1_err_pct"] + 1e-12 for r in val], "o-", label="f1")
        ax[0].loglog([r["elements"] for r in val], [r["defl_err_pct"] + 1e-12 for r in val], "s-", label="deflection")
        ax[0].set_xlabel("elements"); ax[0].set_ylabel("error vs closed form, %"); ax[0].legend(); ax[0].set_title("Beam FE validation")
        names = [r["material"] for r in rows]
        ax[1].bar(range(4), [r["f1_over_speed"] for r in rows]); ax[1].axhline(LIM["f1_ratio"], color="k", ls="--")
        ax[1].set_xticks(range(4)); ax[1].set_xticklabels([n.split()[0] + "\n" + " ".join(n.split()[1:]) for n in names], fontsize=7)
        ax[1].set_ylabel("first natural frequency / shaft speed"); ax[1].set_title("Dynamic margin (dashed = assumed limit)")
        fig.tight_layout(); fig.savefig(OUT / "study.png", dpi=130)
    except Exception as ex:
        print("plot skipped", ex)
    return summary

if __name__ == "__main__":
    s = main()
    print(json.dumps(s, indent=1))
