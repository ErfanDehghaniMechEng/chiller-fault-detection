"""
Chiller Fault Detection with Machine Learning (simulated data)
==============================================================
Binary classification: "normal" (0) vs "faulty" (1).

NOTE: The data below is SIMULATED with simplified physics + noise.
It is meant for teaching ML concepts (class imbalance, Recall vs Precision,
alert threshold), NOT for claiming performance on a real chiller.
To use real data (e.g. ASHRAE RP-1043-based), replace `simulate_chiller_data`
with `pd.read_csv(...)` and keep the rest.

The physics is deliberately simplified and NOT calibrated to a specific chiller
or refrigerant; it only needs to produce realistic-looking relationships.

Outputs (saved in the folder you run the script from):
  fig1_scatter.png      - where do faults appear?
  fig2_correlation.png  - which measurements move together?
  fig3_precision_recall.png - how far can we trust the alerts?
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_predict, train_test_split

RNG = np.random.default_rng(42)
OUT_DIR = "."


# --------------------------------------------------------------------------
# 1) Simulate data
# --------------------------------------------------------------------------
def simulate_chiller_data(n=10_000, fault_rate=0.08):
    load = RNG.uniform(0.35, 1.0, n)               # cooling load fraction
    t_chw_supply = RNG.normal(6.7, 0.3, n)         # chilled water out (C)
    t_cw_in = RNG.uniform(22, 32, n)               # condenser water in (C)
    flow_chw = RNG.normal(1.0, 0.03, n)            # relative flows
    flow_cw = RNG.normal(1.0, 0.03, n)

    # fault labels
    fault_types = np.array(["none", "condenser_fouling", "refrigerant_leak",
                            "low_evap_flow", "low_cond_flow"])
    is_fault = RNG.random(n) < fault_rate
    ftype = np.where(is_fault, RNG.choice(fault_types[1:], n), "none")
    severity = np.where(is_fault, RNG.uniform(0.3, 1.0, n), 0.0)  # mild->strong

    # fault effects
    cond_approach = 3.0 + 4.0 * severity * (ftype == "condenser_fouling")
    evap_approach = 2.0 + 2.5 * severity * (ftype == "refrigerant_leak")
    flow_chw = flow_chw * (1 - 0.35 * severity * (ftype == "low_evap_flow"))
    flow_cw = flow_cw * (1 - 0.35 * severity * (ftype == "low_cond_flow"))
    capacity_loss = 1 - 0.25 * severity * (ftype == "refrigerant_leak")

    # chilled water side
    delta_t_chw = 5.5 * load * capacity_loss / flow_chw
    t_chw_return = t_chw_supply + delta_t_chw
    t_evap = t_chw_supply - evap_approach

    # condenser side
    base_cop = 6.0 - 0.12 * (t_cw_in - 22)
    power = 300 * load / base_cop                  # kW (rough)
    power = power * (1 + 0.5 * (cond_approach - 3) / 4 * 0.4)
    heat_rejected = 300 * load + power
    delta_t_cw = 5.0 * heat_rejected / 360 / flow_cw
    t_cw_out = t_cw_in + delta_t_cw
    t_cond = t_cw_out + cond_approach

    # pressures (kPa), simple linear approximations + sensor noise
    p_evap = 350 + 12 * (t_evap - 5) + RNG.normal(0, 6, n)
    p_cond = 600 + 28 * (t_cond - 35) + RNG.normal(0, 12, n)
    power = power + RNG.normal(0, 4, n)
    t_chw_supply += RNG.normal(0, 0.1, n)
    t_cw_out += RNG.normal(0, 0.15, n)

    return pd.DataFrame({
        "T_chw_supply": t_chw_supply,
        "T_chw_return": t_chw_return,
        "T_cw_in": t_cw_in,
        "T_cw_out": t_cw_out,
        "P_evap": p_evap,
        "P_cond": p_cond,
        "Power_kW": power,
        "Flow_chw": flow_chw,
        "Flow_cw": flow_cw,
        "fault_type": ftype,                       # NOT given to the model
        "faulty": is_fault.astype(int),            # target
    })


df = simulate_chiller_data()
print(f"Records: {len(df)} | Faulty: {df.faulty.mean():.1%}")

# --------------------------------------------------------------------------
# 2) Features / target  (fault_type is dropped: it would leak the answer)
# --------------------------------------------------------------------------
features = [c for c in df.columns if c not in ("fault_type", "faulty")]
X, y = df[features], df["faulty"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)   # test = "unseen exam"

# --------------------------------------------------------------------------
# 3) Model
# --------------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", min_samples_leaf=3,
    random_state=42, n_jobs=-1)

# --------------------------------------------------------------------------
# 4) Choose the alert threshold from training data only (cost-based)
#    Assumption: a missed fault costs 10x more than a false alarm.
# --------------------------------------------------------------------------
COST_MISSED_FAULT = 10
COST_FALSE_ALARM = 1

cv_proba = cross_val_predict(model, X_train, y_train, cv=5,
                             method="predict_proba")[:, 1]
thresholds = np.linspace(0.05, 0.95, 91)
costs = []
for t in thresholds:
    pred = cv_proba >= t
    fn = ((pred == 0) & (y_train == 1)).sum()
    fp = ((pred == 1) & (y_train == 0)).sum()
    costs.append(COST_MISSED_FAULT * fn + COST_FALSE_ALARM * fp)
best_t = thresholds[int(np.argmin(costs))]
print(f"Chosen alert threshold: {best_t:.2f}")

# --------------------------------------------------------------------------
# 5) Final fit + evaluation on unseen test data
# --------------------------------------------------------------------------
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= best_t).astype(int)

recall = recall_score(y_test, pred)
precision = precision_score(y_test, pred)
print(f"Test Recall:    {recall:.1%}  (of real faults, how many we caught)")
print(f"Test Precision: {precision:.1%}  (of alerts, how many were real)")
print(f"Train/test sizes: {len(X_train)}/{len(X_test)}")
print(f"Test faults: {int(y_test.sum())} | Alerts raised: {int(pred.sum())}")

# --------------------------------------------------------------------------
# 6) Figures
# --------------------------------------------------------------------------
GREEN, RED = "#2e9e5b", "#d6392f"

# Fig 1: scatter of two measurements, colored by status
fig, ax = plt.subplots(figsize=(7, 5))
normal, faulty = df[df.faulty == 0], df[df.faulty == 1]
ax.scatter(normal.P_cond, normal.Power_kW, s=6, c=GREEN, alpha=0.35, label="Normal")
ax.scatter(faulty.P_cond, faulty.Power_kW, s=10, c=RED, alpha=0.7, label="Faulty")
ax.set_xlabel("Condenser pressure (kPa)")
ax.set_ylabel("Compressor power (kW)")
ax.set_title("Where do faults appear? (red and green overlap)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig1_scatter.png", dpi=160)
plt.close(fig)

# Fig 2: correlation heatmap (features only)
corr = X.corr()
fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(features)))
ax.set_yticks(range(len(features)))
ax.set_xticklabels(features, rotation=45, ha="right")
ax.set_yticklabels(features)
for i in range(len(features)):
    for j in range(len(features)):
        val = round(float(corr.iloc[i, j]), 1) + 0.0   # +0.0 avoids "-0.0"
        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7,
                color="white" if abs(val) >= 0.6 else "black")
fig.colorbar(im, ax=ax, label="Correlation")
ax.set_title("Which measurements move together?\n(correlation is not causation)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig2_correlation.png", dpi=160)
plt.close(fig)

# Fig 3: precision-recall curve + chosen operating point
p, r, _ = precision_recall_curve(y_test, proba)
ap = average_precision_score(y_test, proba)
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(r, p, color="#1f5fbf", lw=2, label=f"Model (AP = {ap:.2f})")
ax.axhline(y_test.mean(), color="gray", ls="--", lw=1,
           label=f"Random guess ({y_test.mean():.0%} faulty)")
ax.scatter([recall], [precision], c=RED, s=90, zorder=5,
           label=f"Chosen threshold = {best_t:.2f}")
ax.set_xlabel("Recall (of real faults, how many we found)")
ax.set_ylabel("Precision (of alerts, how many were real)")
ax.set_title("How much can we trust the alerts? (unseen test data)")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig3_precision_recall.png", dpi=160)
plt.close(fig)

print("Saved: fig1_scatter.png, fig2_correlation.png, fig3_precision_recall.png")
