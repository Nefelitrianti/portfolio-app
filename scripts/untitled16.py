import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


from pathlib import Path
SCRIPT_DIR = Path(__file__).parent          # scripts/
FILE_PATH = SCRIPT_DIR.parent / "data" / "RMAP_Data_Descriptor_Data.xlsx"
USE_EFA = True

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


if __name__ == "__main__" or not st.session_state.get("_page_config_set"):
    try:
        st.set_page_config(page_title="Global Social Impact Dashboard", layout="wide")
        st.session_state["_page_config_set"] = True
    except Exception:
        pass



st.markdown("""
<style>
/* ── App background ── */
html, body,
[data-testid="stAppViewContainer"],
.main { background: lightgray !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #1a2027 !important;
    font-size: 16px !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
section[data-testid="stSidebar"] *:not(.hypothesis-box):not(.hypothesis-box *) {
    color: #e8eef6 !important;
}

/* ── Hypothesis box (light) ── */
.hypothesis-box {
    background: #f0f7ff !important;
    border-left: 5px solid #3498db !important;
    padding: 14px !important;
    border-radius: 10px !important;
    margin: 14px 0 !important;
    font-size: 16px !important;
    line-height: 1.5 !important;
}
.hypothesis-box, .hypothesis-box * { color: #1f2d3d !important; }
.h0-label  { color: #e74c3c !important; font-weight: 900 !important; }
.h1-label  { color: #27ae60 !important; font-weight: 900 !important; }
.hypothesis-callout {
    background: #f8fbff !important;
    border-left: 4px solid #1f77ff !important;
    padding: 10px !important;
    border-radius: 8px !important;
    margin-top: 10px !important;
}
.result-label { color: #f39c12 !important; font-weight: 900 !important; }

/* ══════════════════════════════════════════
   KPI CARDS
   ══════════════════════════════════════════ */
.kpi-wrap {
    position: relative;
    margin-bottom: 16px;
}
.kpi-card {
    background: #ffffff;
    border: 2px solid #d8dee7;
    border-radius: 18px;
    padding: 52px 18px 18px 18px;
    min-height: 170px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    box-sizing: border-box;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.kpi-card.active {
    border: 2px solid #e74c3c !important;
    box-shadow: 0 0 0 3px rgba(231,76,60,0.12), 0 8px 22px rgba(0,0,0,0.14) !important;
}
.kpi-card-title { font-size: 18px; font-weight: 800; color: #111827; }
.kpi-card-bf    { font-size: 13px; font-weight: 700; color: #374151; }
.kpi-card-def   { font-size: 12px; color: #6b7280; line-height: 1.4; }

/* The button sits inside .kpi-wrap — pull it to top-left over the card */
.kpi-wrap [data-testid="stButton"] {
    position: absolute !important;
    top: 12px !important;
    left: 14px !important;
    z-index: 10 !important;
}
.kpi-wrap [data-testid="stButton"] button {
    padding: 4px 13px !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.6px !important;
    text-transform: uppercase !important;
    border-radius: 20px !important;
    height: auto !important;
    min-height: unset !important;
    line-height: 1.5 !important;
    white-space: nowrap !important;
    box-shadow: none !important;
    transition: all 0.15s !important;
}
.kpi-wrap [data-testid="stButton"] button[kind="secondary"] {
    background: #f1f3f5 !important;
    border: 2px solid #d0d5dd !important;
    color: #374151 !important;
}
.kpi-wrap [data-testid="stButton"] button[kind="secondary"]:hover {
    background: #e74c3c !important;
    border-color: #e74c3c !important;
    color: #fff !important;
}
.kpi-wrap [data-testid="stButton"] button[kind="primary"] {
    background: #e74c3c !important;
    border: 2px solid #e74c3c !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


if "current_factor" not in st.session_state:
    st.session_state.current_factor = "Flexibility"

def set_factor(f):
    st.session_state.current_factor = f
    st.rerun()

@st.cache_data(show_spinner=False)
def load_excel_fast(path: str):
    df0 = pd.read_excel(path, sheet_name=SHEET_NAME, engine="openpyxl")
    df0.columns = clean_names(df0.columns)
    df0.insert(0, "row_id", np.arange(1, len(df0) + 1))

    needed_prefix = "please_rate_the_following_statements"
    likert_cols = [c for c in df0.columns if c.startswith(needed_prefix)][:8]

    edu_col    = "please_share_the_following_total_years_of_full_time_education_from_primary_school_to_higher_education"
    age_col    = "please_share_the_following_your_age_in_years"
    gender_col = "what_is_your_gender"
    eth_col    = "ethnicity_simplified"

    rem_col, off_col = find_hours_cols(df0.columns)

    keep = ["row_id"] + likert_cols
    for c in [edu_col, age_col, gender_col, eth_col, rem_col, off_col]:
        if c is not None and c in df0.columns and c not in keep:
            keep.append(c)
    for i in range(1, 5):
        c = f"Factor{i}_score"
        if c in df0.columns and c not in keep:
            keep.append(c)

    return df0[keep].copy(), likert_cols, rem_col, off_col, edu_col, age_col, gender_col, eth_col

@st.cache_data(show_spinner=False)
def compute_efa_scores(likert_df_numeric: pd.DataFrame):
    from factor_analyzer import FactorAnalyzer
    fa = FactorAnalyzer(n_factors=4, rotation="oblimin", method="minres")
    fa.fit(likert_df_numeric)
    return fa.transform(likert_df_numeric)


st.markdown(
    "<h2 style='text-align:center;margin-bottom:0.4rem;color:white;'>"
    "Remote Work Factor Analysis Dashboard</h2>",
    unsafe_allow_html=True,
)

if not os.path.exists(FILE_PATH):
    st.error("Excel not found. Fix FILE_PATH.")
    st.code(FILE_PATH)
    st.stop()

df, likert_cols, rem_col, off_col, edu_col, age_col, gender_col, eth_col = load_excel_fast(FILE_PATH)


for i in range(1, 5):
    if f"Factor{i}_score" not in df.columns:
        df[f"Factor{i}_score"] = np.nan

if USE_EFA and len(likert_cols) == 8:
    try:
        tmp = df[["row_id"] + likert_cols].copy()
        for c in likert_cols:
            tmp[c] = tmp[c].map(LIKERT_MAP)
        tmp = tmp.dropna()
        if len(tmp) > 30:
            X = tmp[likert_cols].astype(float)
            sc = compute_efa_scores(X)
            sc = pd.DataFrame(sc, columns=[f"Factor{i}_score" for i in range(1, 5)])
            sc["row_id"] = tmp["row_id"].values
            df = df.drop(columns=[f"Factor{i}_score" for i in range(1, 5)], errors="ignore")
            df = df.merge(sc, on="row_id", how="left")
    except Exception:
        pass

for i in range(1, 5):
    df[f"Factor{i}_score"] = pd.to_numeric(df[f"Factor{i}_score"], errors="coerce")


df_clean = df.copy()
df_clean["edu_years"] = to_num(df_clean[edu_col]) if edu_col in df_clean.columns else np.nan
df_clean["age_num"]   = pd.to_numeric(df_clean[age_col], errors="coerce") if age_col in df_clean.columns else np.nan
df_clean["age_group"] = pd.cut(df_clean["age_num"], bins=[-np.inf, 30, 50, np.inf], labels=["Young", "Mid", "Senior"])
df_clean["gender_f"]  = df_clean.get(gender_col, np.nan)
df_clean["gender_f"]  = df_clean["gender_f"].where(df_clean["gender_f"].isin(["Female", "Male"]), np.nan)

if eth_col in df_clean.columns:
    df_clean["ethnicity_f"] = df_clean[eth_col].where(
        df_clean[eth_col].isin(["Asian", "Black", "Mixed", "Other", "White"]), np.nan
    )
else:
    df_clean["ethnicity_f"] = np.nan

if rem_col is None or off_col is None:
    st.error("Remote/office hours columns not found.")
    st.stop()

df_clean["rem_h"]      = pd.to_numeric(df_clean[rem_col], errors="coerce")
df_clean["off_h"]      = pd.to_numeric(df_clean[off_col], errors="coerce")
df_clean["total_h"]    = df_clean["rem_h"] + df_clean["off_h"]
df_clean["pct_remote"] = np.where(df_clean["total_h"] > 0, (df_clean["rem_h"] / df_clean["total_h"]) * 100, np.nan)
df_clean["work_mode"]  = np.select(
    [df_clean["pct_remote"] > 60, (df_clean["pct_remote"] >= 40) & (df_clean["pct_remote"] <= 60), df_clean["pct_remote"] < 40],
    ["Remote", "Hybrid", "Office"],
    default=np.nan
)
work_order = ["Office", "Hybrid", "Remote"]
df_clean["work_mode"] = pd.Categorical(df_clean["work_mode"], categories=work_order, ordered=True)
df_master = df_clean.dropna(subset=["work_mode"]).copy()

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

        # Real button — CSS floats it to top-left of .kpi-wrap
        if st.button(btn_lbl, key=f"kpi_{i}", type=btn_type):
            set_factor(label)

        # Card content sits below, but button overlays via absolute positioning
        st.markdown(f"""
        <div class="{card_cls}">
          <div class="kpi-card-title">{label}</div>
          <div class="kpi-card-bf">BF: {bf_val}</div>
          <div class="kpi-card-def">{defn}</div>
        </div>
        </div>
        """, unsafe_allow_html=True)


f_label    = st.session_state.current_factor
factor_col = factor_to_col[f_label]
group_label = demo_label_map[demo_var]

plot_df = df_master.dropna(subset=[demo_var, factor_col]).copy()
st.markdown("<div style='margin-top: 60px'></div>", unsafe_allow_html=True)
if plot_df.empty:
    st.warning("No data to plot. Enable USE_EFA=True and install factor_analyzer, or ensure factor scores are in the Excel.")
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
    fig.for_each_annotation(lambda a: a.update(
        text=a.text.split("=")[-1].strip()
    ))
    fig.update_layout(
        height=720,
        showlegend=False,
        yaxis_title="Mean Standardized Score",
        xaxis_title="",
        font=dict(size=14),
        margin=dict(l=20, r=20, t=60, b=20),
        title={
            "x":0.5,
            "xanchor":"center"
        }   
    )
    fig.add_hline(y=0, line_dash="dash", opacity=0.35)
    st.plotly_chart(fig, use_container_width=True)
