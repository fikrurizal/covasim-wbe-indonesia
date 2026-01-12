"""
Covasim Simulation for WBE Indonesia (DIY 2021)

How to run (example):
    python run_covasim_diy.py

Dependencies (pin if you want):
    pip install covasim pandas numpy statsmodels matplotlib

Notes on reproducibility:
- Covasim simulations are stochastic. This script sets rand_seed for more stable results.
- Data source: INACOVID GitHub repository (CSV fetched via URL).

Outputs saved to ./outputs/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import covasim as cv
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -----------------------------
# Settings (edit if needed)
# -----------------------------
OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)

INACOVID_URL = "https://raw.githubusercontent.com/erlange/INACOVID/master/data/csv/ext.prov.csv"
LOCATION = "DAERAH ISTIMEWA YOGYAKARTA"
START_DATE = "2021-06-01"
END_DATE = "2021-12-31"

# Scale ~ to 100k population (your original logic: 4.0M / 40 = 100k)
POPULATION = 4_000_000
SCALE_FACTOR = 40

# Sim settings
BASE_SEED = 12345
N_RUNS_BETA = 20
N_RUNS_SWEEP = 10
N_RUNS_PRED = 20

# -----------------------------
# 1) Load and preprocess data
# -----------------------------
data = pd.read_csv(INACOVID_URL)
data["Date"] = pd.to_datetime(data["Date"])

data_diy = data.loc[data["Location"] == LOCATION].copy()
data_diy = data_diy.loc[(data_diy["Date"] >= START_DATE) & (data_diy["Date"] <= END_DATE)]

# Keep/rename the pieces you use
data_diy = (
    data_diy[["Date", "KASUS", "MENINGGAL", "SEMBUH"]]
    .rename(columns={"Date": "date", "KASUS": "new_diagnoses", "MENINGGAL": "new_deaths", "SEMBUH": "new_recoveries"})
    .reset_index(drop=True)
)

# Scale
data_diy_scaled = data_diy.copy()
for col in ["new_deaths", "new_diagnoses", "new_recoveries"]:
    data_diy_scaled[col] = data_diy_scaled[col] / SCALE_FACTOR

datafile = OUTDIR / "data_diy_scaled.csv"
data_diy_scaled.to_csv(datafile, index=False)

# -----------------------------
# 2) Base parameters & interventions
# -----------------------------
pars_b = dict(
    pop_size=POPULATION / SCALE_FACTOR,
    pop_infected=1500,
    beta=0.0126,
    use_waning=True,
    pop_type="hybrid",
    location="Indonesia",
    start_day=START_DATE,
    end_day="2021-10-30",  # keep your original end-day
    rel_severe_prob=0.75,
    rel_crit_prob=0.75,
    rel_death_prob=0.75,
)

im = cv.prior_immunity(60, 0.30)
ct = cv.contact_tracing(trace_probs=dict(h=0.75, s=0.75, w=0.5, c=0.25), do_plot=False)

tpb = cv.test_prob(symp_prob=0.0055, asymp_prob=0.00055, symp_quar_prob=0.055, asymp_quar_prob=0.055, do_plot=False)
tp_50p = cv.test_prob(symp_prob=0.00275, asymp_prob=0.000275, symp_quar_prob=0.0275, asymp_quar_prob=0.0275, do_plot=False)

# -----------------------------
# 3) Baseline run + plot
# -----------------------------
sim_1 = cv.Sim(pars=pars_b, interventions=[tpb, im, ct], label="Baseline", datafile=str(datafile))
sim_1["rand_seed"] = BASE_SEED
sim_1.run()

fig = sim_1.plot(
    style="ggplot",
    to_plot=["new_diagnoses", "new_deaths", "cum_diagnoses", "cum_deaths"],
    do_show=False,
)
fig.savefig(OUTDIR / "baseline_fit.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# -----------------------------
# Helpers
# -----------------------------
def run_scenarios(pars, beta_scenarios, test_prob, n_runs, seed_offset=0):
    summaries = {}
    for s_idx, (scenario, beta_int) in enumerate(beta_scenarios.items()):
        sims = []
        for i in range(n_runs):
            s = cv.Sim(pars=pars, interventions=[test_prob, im, ct, beta_int])
            s["rand_seed"] = int(BASE_SEED + seed_offset + s_idx * 1000 + i)
            sims.append(s)
        msim = cv.MultiSim(sims)
        msim.run()
        msim.mean()
        summaries[scenario] = msim.summary
    return pd.DataFrame({k: v for k, v in summaries.items()})

# -----------------------------
# 4) Beta reduction scenarios (baseline & areas with 50% testing rate)
# -----------------------------
beta_date = "2021-06-01"
beta_scenarios = {f"cb_{p}p": cv.change_beta(days=beta_date, changes=1 - p / 100) for p in [20, 15, 10, 5, 0]}

df_beta_base = run_scenarios(pars_b, beta_scenarios, tpb, n_runs=N_RUNS_BETA, seed_offset=10_000)
df_beta_half = run_scenarios(pars_b, beta_scenarios, tp_50p, n_runs=N_RUNS_BETA, seed_offset=20_000)

df_beta_base.to_csv(OUTDIR / "Beta_Simulation_Baseline.csv")
df_beta_half.to_csv(OUTDIR / "Beta_Simulation_50p_Testing.csv")

# -----------------------------
# 5) Sweep testing probability (baseline beta)
# -----------------------------
symp_prob_values = np.arange(0, 0.060, 0.001)
sweep_summaries = {}

for j, symp_prob in enumerate(symp_prob_values):
    test_prob = cv.test_prob(
        symp_prob=float(symp_prob),
        asymp_prob=float(symp_prob) * 0.1,
        symp_quar_prob=float(symp_prob) * 10,
        asymp_quar_prob=float(symp_prob) * 10,
        do_plot=False,
    )
    sims = []
    for i in range(N_RUNS_SWEEP):
        s = cv.Sim(pars=pars_b, interventions=[test_prob, ct, im])
        s["rand_seed"] = int(BASE_SEED + 30_000 + j * 1000 + i)
        sims.append(s)
    msim = cv.MultiSim(sims)
    msim.run()
    msim.mean()
    sweep_summaries[float(symp_prob)] = msim.summary

df_sweep = pd.DataFrame({k: v for k, v in sweep_summaries.items()})
df_sweep.to_csv(OUTDIR / "Testing_Probability_Simulation.csv")

# -----------------------------
# 6) Quadratic regression: symp_prob ~ f(cum_infections)
# -----------------------------
cum_inf = df_sweep.loc["cum_infections"].astype(float).values
symp_probs = np.array(list(df_sweep.columns), dtype=float)

X = np.column_stack([cum_inf**2, cum_inf, np.ones_like(cum_inf)])
model = sm.OLS(symp_probs, X).fit()

# Plot fit
order = np.argsort(cum_inf)
cum_sorted = cum_inf[order]
X_sorted = np.column_stack([cum_sorted**2, cum_sorted, np.ones_like(cum_sorted)])

plt.figure()
plt.scatter(cum_inf, symp_probs, label="Sweep results")
plt.plot(cum_sorted, model.predict(X_sorted), label="Quadratic fit")
plt.xlabel("Cumulative infections")
plt.ylabel("Testing probability (symp_prob)")
plt.legend()
plt.savefig(OUTDIR / "quadratic_fit.png", dpi=200, bbox_inches="tight")
plt.close()

with open(OUTDIR / "quadratic_model_summary.txt", "w", encoding="utf-8") as f:
    f.write(model.summary().as_text())

# -----------------------------
# 7) Predict testing probs matching beta reduction targets
# -----------------------------
beta_base = pd.read_csv(OUTDIR / "Beta_Simulation_Baseline.csv")
beta_base = beta_base.rename(columns={"Unnamed: 0": "Metric"}) if "Unnamed: 0" in beta_base.columns else beta_base

targets = beta_base.loc[beta_base["Metric"] == "cum_infections", ["cb_20p", "cb_15p", "cb_10p", "cb_5p"]].values.flatten().astype(float)

Xp = np.column_stack([targets**2, targets, np.ones_like(targets)])
pred_probs = model.predict(Xp)

pred_df = pd.DataFrame({"cum_infections": targets, "predicted_symp_prob": pred_probs})
pred_df.to_csv(OUTDIR / "Predicted_Symp_Prob.csv", index=False)

# -----------------------------
# 8) Re-run sims at predicted testing probs (no beta change)
# -----------------------------
pred_summaries = {}
for k, symp_prob in enumerate(pred_probs):
    test_prob = cv.test_prob(
        symp_prob=float(symp_prob),
        asymp_prob=float(symp_prob) * 0.1,
        symp_quar_prob=float(symp_prob) * 10,
        asymp_quar_prob=float(symp_prob) * 10,
        do_plot=False,
    )
    sims = []
    for i in range(N_RUNS_PRED):
        s = cv.Sim(pars=pars_b, interventions=[ct, im, test_prob])
        s["rand_seed"] = int(BASE_SEED + 40_000 + k * 1000 + i)
        sims.append(s)
    msim = cv.MultiSim(sims)
    msim.run()
    msim.mean()
    pred_summaries[float(symp_prob)] = msim.summary

df_pred = pd.DataFrame({k: v for k, v in pred_summaries.items()})
df_pred.to_csv(OUTDIR / "Testing_Probability_Simulation_Prediction.csv")

print("Done. Outputs written to:", OUTDIR.resolve())
