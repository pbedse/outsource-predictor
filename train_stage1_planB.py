import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

df = pd.read_csv("trials_clean.csv")

# ---------------------------------------------------------------
# 1. FEATURES
# ---------------------------------------------------------------
df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce").fillna(0)
df["n_sites"]    = pd.to_numeric(df["n_sites"], errors="coerce").fillna(0)
df["sponsor_trial_count"] = df.groupby("sponsor_final")["nct_id"].transform("count")
df["phase_simple"] = (df["phase"].fillna("NA")
                        .str.replace(";.*", "", regex=True).str.strip().str.upper())

COMPLEX = ["rare","gene","cell therapy","oncology","orphan","sarcoma",
           "leukemia","lymphoma","carcinoma","tumor","tumour"]
df["complex_ta"] = df["conditions"].fillna("").str.lower().apply(
    lambda s: int(any(k in s for k in COMPLEX)))

def pct_rank(s):
    return s.rank(pct=True).fillna(0)

# ---------------------------------------------------------------
# 2. PROPENSITY COMPONENTS  (each scaled 0-1)
# ---------------------------------------------------------------
df["c_sites"]     = pct_rank(df["n_sites"])                    # more sites -> harder to run alone
df["c_enroll"]    = pct_rank(df["enrollment"])                 # bigger trial -> more likely outsourced
df["c_smallspon"] = 1 - pct_rank(df["sponsor_trial_count"])    # less infrastructure -> more likely
df["c_latephase"] = df["phase_simple"].map(
    {"PHASE1":0.30, "PHASE2":0.60, "PHASE3":1.00, "PHASE4":0.50}).fillna(0.40)
df["c_complex"]   = df["complex_ta"].astype(float)

WEIGHTS = {"c_sites":0.35, "c_enroll":0.10, "c_smallspon":0.35,
           "c_latephase":0.15, "c_complex":0.05}
df["outsourcing_score"] = sum(df[c] * w for c, w in WEIGHTS.items())

# ---------------------------------------------------------------
# 3. VALIDATION against trials that DO name a known CRO
# ---------------------------------------------------------------
KNOWN_CROS = ["iqvia","icon plc","parexel","fortrea","medpace","syneos","ppd",
              "pra health","covance","labcorp","thermo fisher","avania",
              "worldwide clinical","scri development","premier research",
              "novotech","emmes","rho inc","veristat","clinipace"]

df["known_cro"] = df["collaborators"].fillna("").str.lower().apply(
    lambda s: int(any(c in s for c in KNOWN_CROS)))

n_pos = int(df["known_cro"].sum())
print("=" * 60)
print("STAGE 1 — OUTSOURCING PROPENSITY SCORE")
print("=" * 60)
print(f"Total trials:                       {len(df)}")
print(f"Trials naming ANY collaborator:     {df['n_collaborators'].gt(0).sum()} "
      f"({df['n_collaborators'].gt(0).mean():.1%})")
print(f"Trials naming a KNOWN CRO:          {n_pos} ({n_pos/len(df):.2%})")
print("\n>> This 1-2% is the core finding: ClinicalTrials.gov does not")
print(">> systematically record who operationally runs a trial.")

if n_pos >= 10:
    auc = roc_auc_score(df["known_cro"], df["outsourcing_score"])
    m1 = df.loc[df.known_cro == 1, "outsourcing_score"].mean()
    m0 = df.loc[df.known_cro == 0, "outsourcing_score"].mean()
    dec = df.nlargest(max(len(df)//10, 1), "outsourcing_score")
    lift = dec["known_cro"].mean() / max(df["known_cro"].mean(), 1e-9)

    print("\n" + "-" * 60)
    print("VALIDATION (against real CRO-labelled trials)")
    print("-" * 60)
    print(f"ROC-AUC:                    {auc:.3f}   (0.50 = random)")
    print(f"Mean score, CRO-named:      {m1:.3f}")
    print(f"Mean score, all others:     {m0:.3f}")
    print(f"Separation:                 {m1-m0:+.3f}")
    print(f"Lift in top decile:         {lift:.2f}x")
else:
    print(f"\n[!] Only {n_pos} positives — too few to validate. Report the")
    print("    score as a transparent heuristic and say so explicitly.")

# ---------------------------------------------------------------
# 4. OUTPUT
# ---------------------------------------------------------------
df.to_csv("trials_scored.csv", index=False)
print("\nSAVED trials_scored.csv")
print("\nTop 10 outsourcing candidates:")
print(df.nlargest(10, "outsourcing_score")[
    ["sponsor_final","phase_simple","enrollment","n_sites","outsourcing_score"]
].to_string(index=False))