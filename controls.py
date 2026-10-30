import json, numpy as np, torch
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from lcjepa import *

test = make_batch(1000, rng=np.random.default_rng(123))
n, W, N = test["t"].shape
rs = np.random.default_rng(0)
star_train = rs.random(n) < 0.7          # same split rule as evaluate()
tok_star = np.repeat(np.arange(n), W * N)
per = (test["cls"] != 3)[tok_star]
tr, te = star_train[tok_star] & per, ~star_train[tok_star] & per
ph = test["phase"].reshape(-1)
Y = np.stack([np.cos(2 * np.pi * ph), np.sin(2 * np.pi * ph)], 1)

def ridge(X):
    sc = StandardScaler().fit(X[tr])
    m = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(sc.transform(X[tr]), Y[tr])
    p = m.predict(sc.transform(X[te]))
    return [r2_score(Y[te][:, k], p[:, k]) for k in range(2)]

def mlp(X):
    i = rs.choice(np.where(tr)[0], 20000, replace=False)
    j = rs.choice(np.where(te)[0], 20000, replace=False)
    sc = StandardScaler().fit(X[i])
    m = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=40, early_stopping=True,
                     random_state=0).fit(sc.transform(X[i]), Y[i])
    p = m.predict(sc.transform(X[j]))
    return [r2_score(Y[j][:, k], p[:, k]) for k in range(2)]

out = {}
# raw control: the observation's own magnitude/error plus simple window statistics
mag = test["mag"]
stats = np.stack([mag.mean(-1), mag.std(-1), mag.min(-1), mag.max(-1)], -1)  # (n, W, 4)
raw = np.concatenate([mag[..., None], test["err"][..., None],
                      np.repeat(stats[:, :, None, :], N, 2)], -1).reshape(n * W * N, -1)
out["raw inputs (own mag + window stats)"] = dict(mlp=mlp(raw))

def model_feats(var, ckpt=True):
    m = Model(var)
    if ckpt:
        m.load_state_dict(torch.load(f"ckpt_{var}.pt"))
    H, S, C = embed_dataset(m, test)
    c_tok = np.repeat(C.reshape(n * W, -1), N, 0)
    return H.reshape(n * W * N, -1), S.reshape(n * W * N, -1), c_tok

torch.manual_seed(0)
H, S, C = model_feats("rot+time", ckpt=False)
out["untrained encoder"] = dict(lin_h=ridge(H), mlp_hc=mlp(np.concatenate([H, C], 1)))
import sys
VARS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["lejepa", "mae", "mlp+batch", "mlp+time", "rot+batch", "rot+time"]
for var in VARS:
    try:
        H, S, C = model_feats(var)
    except FileNotFoundError:
        continue
    d = dict(lin_h=ridge(H), mlp_hc=mlp(np.concatenate([H, C], 1)))
    if not var.startswith(("lejepa", "mae", "mjepa")):
        d["lin_s"] = ridge(S)
        d["mlp_sc"] = mlp(np.concatenate([S, C], 1))
    out[var] = d
    print(var, {k: [round(x, 3) for x in v] for k, v in d.items()}, flush=True)
json.dump(out, open("controls.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:40s}", {kk: [round(x, 3) for x in vv] for kk, vv in v.items()})
