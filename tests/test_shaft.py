import sys, pathlib, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import shaft as s

def test_beam_fe_matches_closed_form():
    for r in s.validation():
        assert r["defl_err_pct"] < 0.01 and r["f1_err_pct"] < 0.01 and r["f3_err_pct"] < 0.05

def test_reactions_balance_statics():
    T, Ft, Fr, Fa = s.loads()
    x = np.array([0.0, s.L])
    assert np.allclose(s.bending_moment(x, Ft, s.X_GEAR), 0.0, atol=1e-9)

def test_min_size_meets_limits_and_is_minimal():
    for name, m in s.MATS.items():
        r = s.min_size(name, m)
        assert r["pass"]
        assert not s.study(name, m, r["diameter_scale"] * 0.98)["pass"]
