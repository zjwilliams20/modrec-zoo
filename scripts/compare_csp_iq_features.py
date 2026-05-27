import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

from modreczoo.data import load_dataset, normalize_signal, iq_features, csp_expert_features

DATASET = "data/baseline_4096"
N_MAX = 5000

print(f"Loading {DATASET}...")
signals, metadata = load_dataset(DATASET)

labels_raw = metadata["modulation"].to_numpy()
le = LabelEncoder()
y = le.fit_transform(labels_raw)
print(f"Classes: {list(le.classes_)}")
print(f"Total signals: {len(signals)}, using {N_MAX}")

rng = np.random.default_rng(42)
idx = rng.choice(len(signals), size=min(N_MAX, len(signals)), replace=False)

print("\nExtracting features...")
iq_feats, csp_feats = [], []
t0 = time.time()
for i in idx:
    x = normalize_signal(signals[i])
    iq_feats.append(iq_features(x))
print(f"  iq_features:  {time.time()-t0:.1f}s ({len(iq_feats[0])} features)")

t0 = time.time()
for i in idx:
    x = normalize_signal(signals[i])
    csp_feats.append(csp_expert_features(x))
print(f"  csp_features: {time.time()-t0:.1f}s ({len(csp_feats[0])} features)")

X_iq  = np.array(iq_feats)
X_csp = np.array(csp_feats)
y_sub = y[idx]

X_iq_tr,  X_iq_te,  y_tr, y_te = train_test_split(X_iq,  y_sub, test_size=0.2, random_state=42, stratify=y_sub)
X_csp_tr, X_csp_te, _,    _    = train_test_split(X_csp, y_sub, test_size=0.2, random_state=42, stratify=y_sub)

print("\n--- Random Forest (100 trees) ---")
for name, Xtr, Xte in [("iq_features ", X_iq_tr, X_iq_te), ("csp_features", X_csp_tr, X_csp_te)]:
    t0 = time.time()
    clf = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)
    clf.fit(Xtr, y_tr)
    acc = accuracy_score(y_te, clf.predict(Xte))
    print(f"  {name}: test_acc={acc:.4f}  ({time.time()-t0:.1f}s)")
    if name.startswith("csp"):
        feats = [f"f{i+1}" for i in range(X_csp_tr.shape[1])]
        importances = clf.feature_importances_
        ranked = sorted(zip(feats, importances), key=lambda x: -x[1])
        print(f"    feature importances: {', '.join(f'{n}={v:.3f}' for n,v in ranked)}")

print("\n--- Logistic Regression ---")
for name, Xtr, Xte in [("iq_features ", X_iq_tr, X_iq_te), ("csp_features", X_csp_tr, X_csp_te)]:
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    t0 = time.time()
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(Xtr_s, y_tr)
    acc = accuracy_score(y_te, clf.predict(Xte_s))
    print(f"  {name}: test_acc={acc:.4f}  ({time.time()-t0:.1f}s)")