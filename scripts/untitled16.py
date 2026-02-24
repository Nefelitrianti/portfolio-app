import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="Global Social Impact Dashboard", layout="wide")

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parent   # repo root
FILE_PATH = BASE_DIR / "data" / "RMAP_Data_Descriptor_Data.xlsx"
SHEET_NAME = "RMAP_Data_Descriptor_Data"

USE_EFA = True

# ---------- SETTINGS ----------
LIKERT_LEVELS = [
    "Strongly Disagree", "Disagree", "Somewhat Disagree",
    "Neither Agree nor Disagree", "Somewhat Agree", "Agree", "Strongly Agree"
]
LIKERT_MAP = {k: i + 1 for i, k in enumerate(LIKERT_LEVELS)}

factor_labels = ["Flexibility", "Challenges", "Career Anxiety", "WLB Struggle"]
factor_to_col = {
    "Flexibility": "Factor1_score",
    "Challenges": "Factor2_score",
    "Career Anxiety": "Factor3_score",
    "WLB Struggle": "Factor4_score",
}
factor_defs = {
    "Flexibility": "Control over work schedule.",
    "Challenges": "Technical/social barriers.",
    "Career Anxiety": "Fear of missing promotions.",
    "WLB Struggle": "Home/work boundaries.",
}
demo_label_map = {"gender_f": "Gender", "age_group": "Age Group", "ethnicity_f": "Ethnicity"}

bf_lookup = {
    "gender_f": {"Flexibility": "1.08e+25", "Challenges": "6.15e+07", "Career Anxiety": "0.08", "WLB Struggle": "3.63"},
    "age_group": {"Flexibility": "1.05e+17", "Challenges": "1.97e+64", "Career Anxiety": "9.53e+13", "WLB Struggle": "1.37e+27"},
    "ethnicity_f": {"Flexibility": "4.21e+05", "Challenges": "1.12e+03", "Career Anxiety": "0.12", "WLB Struggle": "0.95"},
}

# ---------- HELPERS ----------
def clean_names(cols):
    out = []
    for c in cols:
        c = str(c).strip().lower()
        c = re.sub(r"[^a-z0-9]+", "_", c)
        c = re.sub(r"_+", "_", c).strip("_")
        out.append(c)
    return out

def to_num(series):
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def evidence_text_from_bf(bf_val):
    try:
        bf = float(bf_val)
    except:
        return ""
    if bf > 100: return "Extreme evidence (supports H₁)."
    if bf > 30:  return "Very strong evidence (supports H₁)."
    if bf > 10:  return "Strong evidence (supports H₁)."
    if bf > 3:   return "Moderate evidence (supports H₁)."
    if bf < 1:   return "Evidence favors H₀ (no difference)."
    return "Anecdotal / weak evidence."

def make_label(f_label, m):
    if pd.isna(m):
        return ""
    if f_label == "Flexibility":
        return "Less Flexible" if m < 0 else "More Flexible"
    if f_label == "Challenges":
        return "Fewer Barriers" if m < 0 else "More Barriers"
    if f_label == "Career Anxiety":
        return "Less Anxious" if m < 0 else "More Anxious"
    if f_label == "WLB Struggle":
        return "Better WLB" if m < 0 else "Struggling WLB"
    return ""

def find_hours_cols(cols):
    STEM = "on_average_what_amount_of_time_of_your_weekly_work_schedule_do_you_perform_remotely_in_the_office"
    cols = list(cols)
    rem_candidates = [c for c in cols if STEM in c and "remotely" in c and "hours" in c and not c.endswith("in_the_office_hours")]
    off_candidates = [c for c in cols if STEM in c and c.endswith("in_the_office_hours")]
    if not off_candidates:
        off_candidates = [c for c in cols if STEM in c and "in_the_office" in c and "hours" in c]
    if not rem_candidates:
        rem_candidates = [c for c in cols if ("remotely" in c and "hours" in c)]
    if not off_candidates:
        off_candidates = [c for c in cols if ("in_the_office" in c and "hours" in c)]
    return (rem_candidates[0] if rem_candidates else None, off_candidates[0] if off_candidates else None)

def likert_to_num(s: pd.Series) -> pd.Series:
    """
    Robust converter:
    - handles numeric 1..7 / "1"
    - handles "1 - Strongly Disagree"
    - handles exact text labels in LIKERT_MAP
    """
    s_txt = s.astype(str).str.strip()

    s_num = pd.to_numeric(s_txt, errors="coerce")
    s_digit = pd.to_numeric(s_txt.str.extract(r"([1-7])", expand=False), errors="coerce")
    s_map = s_txt.map(LIKERT_MAP)

    return s_num.fillna(s_digit).fillna(s_map)

# ---------- CSS ----------
# Paste your full CSS instead of "..."
st.markdown(""" 
<style>
/* example */
html, body, [data-testid="stAppViewContainer"], .main { background: #0f141a !important; }
</style>
""", unsafe_allow_html=True)

# ---------- STATE ----------
if "current_factor" not in st.session_state:
    st.session_state.current_factor = "Flexibility"

def set_factor(f):
    st.session_state.current_factor = f
    st.rerun()

# ---------- LOAD ----------
@st.cache_data(show_spinner=False)
def load_excel_fast(path: Path):
    df0 = pd.read_excel(path, sheet_name=SHEET_NAME, engine="openpyxl")
    df0.columns = clean_names(df0.columns)
    df0.insert(0, "row_id", np.arange(1, len(df0) + 1))

    # ✅ robust Likert detection for YOUR dataset
    likert_cols = sorted([
        c for c in df0.columns
        if "please_rate_the_following_statements_regarding_remote_work" in c
    ])[:8]

    edu_col    = "please_share_the_following_total_years_of_full_time_education_from_primary_school_to_higher_education"
    age_col    = "please_share_the_following_your_age_in_years"
    gender_col = "what_is_your_gender"
    eth_col    = "ethnicity_simplified"

    rem_col, off_col = find_hours_cols(df0.columns)

    keep = ["row_id"] + likert_cols
    for c in [edu_col, age_col, gender_col, eth_col, rem_col, off_col]:
        if c is not None and c in df0.columns and c not in keep:
            keep.append(c)

    return df0[keep].copy(), likert_cols, rem_col, off_col, edu_col, age_col, gender_col, eth_col

def compute_efa_scores(likert_df_numeric: pd.DataFrame) -> np.ndarray:
    from factor_analyzer import FactorAnalyzer
    fa = FactorAnalyzer(n_factors=4, rotation="oblimin", method="minres")
    fa.fit(likert_df_numeric)
    return fa.transform(likert_df_numeric)

# ---------- TITLE ----------
st.markdown(
    "<h2 style='text-align:center;margin-bottom:0.4rem;color:white;'>"
    "Remote Work Factor Analysis Dashboard</h2>",
    unsafe_allow_html=True,
)

# ---------- FILE CHECK ----------
st.write("BASE_DIR:", BASE_DIR)
st.write("Looking for:", FILE_PATH)
st.write("Exists:", FILE_PATH.exists())

if not FILE_PATH.exists():
    st.error("Excel not found. Fix FILE_PATH.")
    st.code(str(FILE_PATH))
    st.stop()

# ---------- READ ----------
df, likert_cols, rem_col, off_col, edu_col, age_col, gender_col, eth_col = load_excel_fast(FILE_PATH)

st.write("Likert cols found:", len(likert_cols))
st.write(likert_cols)
st.write("rem_col:", rem_col)
st.write("off_col:", off_col)

# Ensure factor cols exist
for i in range(1, 5):
    c = f"Factor{i}_score"
    if c not in df.columns:
        df[c] = np.nan

# ---------- EFA ----------
if USE_EFA and len(likert_cols) >= 4:
    tmp = df[["row_id"] + likert_cols].copy()
    for c in likert_cols:
        tmp[c] = likert_to_num(tmp[c])

    st.write("Usable rows for EFA (before dropna):", len(tmp))
    st.write("Non-null per Likert col:", tmp[likert_cols].notna().sum().to_dict())

    tmp = tmp.dropna(subset=likert_cols)

    st.write("Usable rows for EFA (after dropna):", len(tmp))

    if len(tmp) > 30:
        try:
            X = tmp[likert_cols].astype(float)
            sc = compute_efa_scores(X)
            sc = pd.DataFrame(sc, columns=[f"Factor{i}_score" for i in range(1, 5)])
            sc["row_id"] = tmp["row_id"].values

            df = df.drop(columns=[f"Factor{i}_score" for i in range(1, 5)], errors="ignore")
            df = df.merge(sc, on="row_id", how="left")

        except Exception as e:
            st.error("EFA failed")
            st.exception(e)
    else:
        st.warning("Not enough complete Likert rows (>30) to run EFA.")

# Coerce factor columns numeric
for i in range(1, 5):
    df[f"Factor{i}_score"] = pd.to_numeric(df[f"Factor{i}_score"], errors="coerce")

st.write("Factor non-null counts:", {f"Factor{i}_score": df[f"Factor{i}_score"].notna().sum() for i in range(1, 5)})

# ---------- CLEAN / FEATURES ----------
df_clean = df.copy()
df_clean["edu_years"] = to_num(df_clean[edu_col]) if edu_col in df_clean.columns else np.nan
df_clean["age_num"]   = pd.to_numeric(df_clean[age_col], errors="coerce") if age_col in df_clean.columns else np.nan
df_clean["age_group"] = pd.cut(df_clean["age_num"], bins=[-np.inf, 30, 50, np.inf], labels=["Young", "Mid", "Senior"])

# gender mapping (robust)
g = df_clean.get(gender_col, pd.Series(index=df_clean.index, dtype="object")).astype(str).str.strip().str.lower()
df_clean["gender_f"] = np.select(
    [g.isin(["female", "woman"]), g.isin(["male", "man"])],
    ["Female", "Male"],
    default=pd.NA
)

# ethnicity mapping (keep only main groups; others -> NA)
if eth_col in df_clean.columns:
    e = df_clean[eth_col].astype(str).str.strip()
    df_clean["ethnicity_f"] = e.where(e.isin(["Asian", "Black", "Mixed", "Other", "White"]), pd.NA)
else:
    df_clean["ethnicity_f"] = pd.NA

if rem_col is None or off_col is None:
    st.error("Remote/office hours columns not found.")
    st.stop()

# robust numeric hours (comma decimal safe)
df_clean["rem_h"] = pd.to_numeric(df_clean[rem_col].astype(str).str.replace(",", "."), errors="coerce")
df_clean["off_h"] = pd.to_numeric(df_clean[off_col].astype(str).str.replace(",", "."), errors="coerce")

df_clean["total_h"] = df_clean["rem_h"] + df_clean["off_h"]
df_clean["pct_remote"] = np.where(df_clean["total_h"] > 0, (df_clean["rem_h"] / df_clean["total_h"]) * 100, np.nan)

df_clean["work_mode"] = np.select(
    [
        df_clean["pct_remote"] > 60,
        (df_clean["pct_remote"] >= 40) & (df_clean["pct_remote"] <= 60),
        df_clean["pct_remote"] < 40
    ],
    ["Remote", "Hybrid", "Office"],
    default=""
).astype("object")

df_clean["work_mode"] = df_clean["work_mode"].replace("", pd.NA)
work_order = ["Office", "Hybrid", "Remote"]
df_clean["work_mode"] = pd.Categorical(df_clean["work_mode"], categories=work_order, ordered=True)
df_master = df_clean.dropna(subset=["work_mode"]).copy()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### Analyze Per Demographic Team")
    demo_var = st.selectbox(
        "",
        options=["gender_f", "age_group", "ethnicity_f"],
        format_func=lambda x: demo_label_map[x],
    )
    st.markdown("---")
    st.markdown("### Evidence (BF)")

    factor_name = st.session_state.current_factor
    bf_val_side = bf_lookup.get(demo_var, {}).get(factor_name, "NA")
    st.metric("Bayes Factor", value=str(bf_val_side))
    st.caption(evidence_text_from_bf(bf_val_side))

    group_label = demo_label_map[demo_var]
    try:
        bf_num = float(bf_val_side)
    except:
        bf_num = np.nan

    if np.isnan(bf_num):
        result_txt = "Inconclusive: Not enough data."
    elif bf_num > 100:
        result_txt = "Extreme Support for H₁: The data shows an overwhelming difference. We reject H₀ with high certainty."
    elif bf_num > 3:
        result_txt = "Support for H₁: There is substantial evidence that groups differ."
    elif bf_num < 1:
        result_txt = "Support for H₀: The data suggests these groups are the same."
    else:
        result_txt = "Inconclusive: The data doesn't strongly favor H₀ or H₁."

    st.markdown(f"""
    <div style="color:#e8eef6;font-size:15px;margin-top:10px;">
      In this analysis, we use the Bayes Factor (BF) to choose between:
    </div>
    <div class="hypothesis-box">
      <div style="margin-bottom:10px;">
        <span class="h0-label">H₀ (Null Hypothesis):</span>
        The groups are the same (No effect of demographic).
      </div>
      <div>
        <span class="h1-label">H₁ (Alternative Hypothesis):</span>
        The groups are different (Demographic matters).
      </div>
      <div class="hypothesis-callout">
        <div><span class="h0-label">H₀:</span> No significant difference in <b>{factor_name}</b> across <b>{group_label}</b> groups.</div>
        <div style="margin-top:8px;"><span class="h1-label">H₁:</span> <b>{group_label}</b> significantly influences <b>{factor_name}</b>.</div>
      </div>
      <div style="margin-top:15px;">
        <span class="result-label">Result:</span>
        <span style="color:#1f2d3d;font-weight:500;">{result_txt}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ---------- KPI CARDS ----------
cols = st.columns(4)
for i, label in enumerate(factor_labels):
    bf_val   = bf_lookup.get(demo_var, {}).get(label, "NA")
    defn     = factor_defs[label]
    active   = (st.session_state.current_factor == label)
    card_cls = "kpi-card active" if active else "kpi-card"
    btn_lbl  = "✓ Selected" if active else "Select"
    btn_type = "primary" if active else "secondary"

    with cols[i]:
        st.markdown('<div class="kpi-wrap">', unsafe_allow_html=True)
        if st.button(btn_lbl, key=f"kpi_{i}", type=btn_type):
            set_factor(label)
        st.markdown(f"""
        <div class="{card_cls}">
          <div class="kpi-card-title">{label}</div>
          <div class="kpi-card-bf">BF: {bf_val}</div>
          <div class="kpi-card-def">{defn}</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

# ---------- PLOT ----------
f_label = st.session_state.current_factor
factor_col = factor_to_col[f_label]
group_label = demo_label_map[demo_var]

plot_df = df_master.dropna(subset=[demo_var, factor_col]).copy()

st.markdown("<div style='margin-top: 60px'></div>", unsafe_allow_html=True)

if plot_df.empty:
    st.warning("No data to plot. Check factor scores and demographic filtering.")
else:
    plot_df[demo_var] = plot_df[demo_var].astype("category")

    agg = (
        plot_df
        .groupby(["work_mode", demo_var], as_index=False)[factor_col]
        .mean()
        .rename(columns={factor_col: "mean_score"})
    )

    all_demo_levels = plot_df[demo_var].cat.categories
    full_index = pd.MultiIndex.from_product(
        [pd.Categorical(work_order, categories=work_order, ordered=True), all_demo_levels],
        names=["work_mode", demo_var],
    )

    agg_full = agg.set_index(["work_mode", demo_var]).reindex(full_index).reset_index()
    agg_full["label_text"] = agg_full["mean_score"].apply(lambda m: make_label(f_label, m))

    fig = px.bar(
        agg_full,
        x="work_mode",
        y="mean_score",
        color="work_mode",
        facet_col=demo_var,
        text="label_text",
        title=f"Analysis of {f_label} Grouped by {group_label}",
        category_orders={"work_mode": work_order},
    )
    fig.update_traces(textposition="outside")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].strip()))
    fig.update_layout(
        height=720,
        showlegend=False,
        yaxis_title="Mean Standardized Score",
        xaxis_title="",
        font=dict(size=14),
        margin=dict(l=20, r=20, t=60, b=20),
        title={"x": 0.5, "xanchor": "center"},
    )
    fig.add_hline(y=0, line_dash="dash", opacity=0.35)
    st.plotly_chart(fig, use_container_width=True)
