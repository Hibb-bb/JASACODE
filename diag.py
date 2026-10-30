import sys, json, numpy as np, torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from lcjepa import *

test = make_batch(1000, rng=np.random.default_rng(123))
n = test["t"].shape[0]
tr = np.random.default_rng(0).random(n) < 0.7
per = test["cls"] != 3
targets = {"noise level sigma": test["err"][:, 0, 0], "amplitude A": test["A"],
           "log P (periodic)": test["logP"]}

def r2(X, y, mask):
    a, b = tr & mask, ~tr & mask
    sc = StandardScaler().fit(X[a])
    m = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(sc.transform(X[a]), y[a])
    return r2_score(y[b], m.predict(sc.transform(X[b])))

out = {}
VARS = sys.argv[1].split(",")
for var in VARS:
    m = Model(var); m.load_state_dict(torch.load(f"ckpt_{var}.pt")); m.eval()
    H, S, C = embed_dataset(m, test)
    ca = C[:, 0]
    res = {}
    for name, y in targets.items():
        mask = per if "log P" in name else np.ones(n, bool)
        res[name] = round(r2(ca, y, mask), 3)
    # how many SIGReg-isotropic directions of c are explained by (sigma, A) alone?
    sa = np.stack([test["err"][:, 0, 0], test["A"]], 1)
    feats = np.concatenate([sa, sa ** 2, sa[:, :1] * sa[:, 1:]], 1)
    sc = StandardScaler().fit(feats[tr])
    rr = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(sc.transform(feats[tr]), ca[tr])
    res["share of c variance explained by (sigma, A)"] = round(
        r2_score(ca[~tr], rr.predict(sc.transform(feats[~tr])), multioutput="variance_weighted"), 3)
    if var.startswith("rot"):
        # per-harmonic prediction error over 0.5-5 day gaps, periodic stars where LS found f
        bt = to_torch(test, "cpu")
        with torch.no_grad():
            _, s, c = m.encode_batch(bt)
            w_ls = window_w_ls(bt)
            i, j, dt, mask = sample_pairs(bt["t"], 256, 5.0, dt_min=0.5)
            B, W, N, ds = s.shape
            sf = s.reshape(B, W * N, ds)
            si = sf.gather(1, i[..., None].expand(-1, -1, ds)).reshape(-1, ds)
            sj = sf.gather(1, j[..., None].expand(-1, -1, ds)).reshape(-1, ds)
            cs = c.gather(1, (i // N)[..., None].expand(-1, -1, c.shape[-1])).reshape(-1, c.shape[-1])
            ws = w_ls.gather(1, i // N).reshape(-1)
            p = m.pred(si, cs, dt.reshape(-1), ws)
        ratio = (w_ls[:, 0].numpy() / (2 * np.pi / 10 ** np.where(per, test["logP"], 0)))
        good = per & (np.abs(ratio - 1) < 0.02)
        keep = (mask & torch.as_tensor(good)[:, None]).reshape(-1)
        e = (p - sj)[keep].pow(2).mean(0).view(-1, 2).mean(1).numpy()
        var_t = sj[keep].pow(2).mean(0).view(-1, 2).mean(1).numpy()
        res["pred error / target power, per block k (speed k/2 * w_LS)"] = [
            round(float(a / b), 2) for a, b in zip(e, var_t)]
    out[var] = res
    print(var, res, flush=True)
json.dump(out, open("diag_" + "_".join(v.replace("+", "") for v in VARS) + ".json", "w"), indent=1)
