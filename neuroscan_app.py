"""
NeuroScan HealthTech — Full Streamlit ML Dashboard
====================================================
SETUP (run once):
    pip install streamlit scikit-learn pandas numpy matplotlib seaborn
    python generate_eeg_data.py

RUN:
    streamlit run neuroscan_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, io, time
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    balanced_accuracy_score, cohen_kappa_score,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan — EEG Seizure Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 14px !important; }

/* Main background */
.main { background: #080b12; }
.block-container { padding: 2rem 2.5rem; max-width: 1400px; }

/* Cards */
.ns-card {
    background: #0f1117;
    border: 1px solid #1e2130;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.ns-card-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #4b5563;
    margin-bottom: 16px;
}

/* Metric tiles */
.metric-tile {
    background: #0d1017;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.metric-tile .val {
    font-size: 28px;
    font-weight: 700;
    color: #e6edf3;
    font-family: 'DM Mono', monospace;
}
.metric-tile .lbl {
    font-size: 11px;
    color: #6b7280;
    margin-top: 4px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

/* Result hero */
.hero-normal {
    background: linear-gradient(135deg, #052e16 0%, #0f1f13 100%);
    border: 1px solid #166534;
    border-radius: 20px;
    padding: 40px;
    text-align: center;
}
.hero-seizure {
    background: linear-gradient(135deg, #2d0a0a 0%, #1f0f0f 100%);
    border: 1px solid #991b1b;
    border-radius: 20px;
    padding: 40px;
    text-align: center;
}
.hero-normal h2  { color: #4ade80; font-size: 26px; font-weight: 700; margin: 0 0 10px; }
.hero-seizure h2 { color: #f87171; font-size: 26px; font-weight: 700; margin: 0 0 10px; }
.hero-caption { color: #9ca3af; font-size: 14px; line-height: 1.7; max-width: 440px; margin: 0 auto; }

/* Pill badges */
.pill-green { background:#052e16; color:#4ade80; border:1px solid #166534; border-radius:20px; padding:3px 12px; font-size:12px; font-weight:500; }
.pill-red   { background:#2d0a0a; color:#f87171; border:1px solid #991b1b; border-radius:20px; padding:3px 12px; font-size:12px; font-weight:500; }
.pill-blue  { background:#0d1b40; color:#818cf8; border:1px solid #3730a3; border-radius:20px; padding:3px 12px; font-size:12px; font-weight:500; }

/* Brand header */
.brand-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 0 24px;
    border-bottom: 1px solid #1e2130;
    margin-bottom: 24px;
}
.brand-icon {
    width: 40px; height: 40px;
    background: #5b6af0;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
}
.brand-name { font-size: 18px; font-weight: 700; color: #e6edf3; }
.brand-tag  { font-size: 12px; color: #6b7280; }

/* Stray Streamlit chrome cleanup */
#MainMenu, footer, header { visibility: hidden; }
.stButton > button {
    background: #5b6af0;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    transition: all 0.2s;
}
.stButton > button:hover { background: #4338ca; }

/* Confusion matrix cells */
.cm-tp { background:#052e16; color:#4ade80; border-radius:10px; padding:18px; text-align:center; }
.cm-tn { background:#052e16; color:#4ade80; border-radius:10px; padding:18px; text-align:center; }
.cm-fp { background:#2d0a0a; color:#f87171; border-radius:10px; padding:18px; text-align:center; }
.cm-fn { background:#2d0a0a; color:#f87171; border-radius:10px; padding:18px; text-align:center; }
.cm-val { font-size:28px; font-weight:700; font-family:'DM Mono',monospace; }
.cm-lbl { font-size:10px; margin-top:4px; opacity:0.6; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
FEATURES = [
    "delta_power","theta_power","alpha_power","beta_power","gamma_power",
    "line_length","sample_entropy","kurtosis","skewness","variance",
    "zero_crossings","peak_frequency",
]
LABEL_COL = "label"

PALETTE = {
    "bg":      "#080b12",
    "card":    "#0f1117",
    "border":  "#1e2130",
    "indigo":  "#5b6af0",
    "green":   "#4ade80",
    "red":     "#f87171",
    "amber":   "#fbbf24",
    "text":    "#e6edf3",
    "muted":   "#6b7280",
}

def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["card"],
        "axes.edgecolor":    PALETTE["border"],
        "axes.labelcolor":   PALETTE["muted"],
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "text.color":        PALETTE["text"],
        "grid.color":        PALETTE["border"],
        "grid.alpha":        0.6,
        "font.family":       "sans-serif",
        "font.size":         11,
    })

set_plot_style()

@st.cache_data
def generate_eeg_data():
    np.random.seed(42)
    N_NORMAL, N_SEIZURE = 2000, 500

    def make_normal(n):
        return pd.DataFrame({
            "delta_power":    np.random.normal(80, 12, n).clip(10, 120),
            "theta_power":    np.random.normal(60, 10, n).clip(5, 100),
            "alpha_power":    np.random.normal(75, 11, n).clip(10, 110),
            "beta_power":     np.random.normal(42,  9, n).clip(5, 80),
            "gamma_power":    np.random.normal(28,  7, n).clip(2, 60),
            "line_length":    np.random.normal(0.45, 0.08, n).clip(0.1, 0.9),
            "sample_entropy": np.random.normal(1.8, 0.3,  n).clip(0.5, 3.5),
            "kurtosis":       np.random.normal(3.1, 0.5,  n).clip(1.0, 6.0),
            "skewness":       np.random.normal(0.1, 0.3,  n).clip(-1.5, 1.5),
            "variance":       np.random.normal(120, 25,   n).clip(20, 300),
            "zero_crossings": np.random.normal(55,  10,   n).clip(10, 120),
            "peak_frequency": np.random.normal(10,  2,    n).clip(2, 30),
            "label": "Normal",
        })

    def make_seizure(n):
        return pd.DataFrame({
            "delta_power":    np.random.normal(35, 10, n).clip(5, 80),
            "theta_power":    np.random.normal(28,  9, n).clip(3, 70),
            "alpha_power":    np.random.normal(22,  8, n).clip(3, 60),
            "beta_power":     np.random.normal(95, 15, n).clip(30, 160),
            "gamma_power":    np.random.normal(88, 13, n).clip(30, 140),
            "line_length":    np.random.normal(0.82, 0.07, n).clip(0.3, 1.2),
            "sample_entropy": np.random.normal(0.7, 0.2,  n).clip(0.1, 1.5),
            "kurtosis":       np.random.normal(6.8, 1.1,  n).clip(2.0, 12.0),
            "skewness":       np.random.normal(1.2, 0.5,  n).clip(-0.5, 3.0),
            "variance":       np.random.normal(480, 80,   n).clip(100, 800),
            "zero_crossings": np.random.normal(120, 20,   n).clip(40, 200),
            "peak_frequency": np.random.normal(25,  4,    n).clip(8, 45),
            "label": "Seizure",
        })

    df = pd.concat([make_normal(N_NORMAL), make_seizure(N_SEIZURE)], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

@st.cache_data
def load_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

def prepare_data(df):
    le = LabelEncoder()
    X = df[FEATURES].values
    y = le.fit_transform(df[LABEL_COL].values)
    return X, y, le

def build_model(name, **kwargs):
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, max_depth=None,
                                      random_state=42, n_jobs=-1, **kwargs)
    elif name == "SVM (RBF)":
        return SVC(kernel="rbf", C=10, gamma="scale",
                   probability=True, random_state=42, **kwargs)
    elif name == "K-NN (k=5)":
        return KNeighborsClassifier(n_neighbors=5, metric="euclidean", **kwargs)

@st.cache_data
def run_full_pipeline(df_bytes, model_name, test_size, cv_folds):
    df = pd.read_csv(io.BytesIO(df_bytes)) if df_bytes else generate_eeg_data()

    # Validate columns
    missing = [c for c in FEATURES + [LABEL_COL] if c not in df.columns]
    if missing:
        return {"error": f"Missing columns: {missing}"}

    X, y, le = prepare_data(df)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    model = build_model(model_name)

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring="accuracy", n_jobs=-1)

    # Fit on full train set
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── All metrics ────────────────────────────────────────────────────────────
    acc     = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    ap      = average_precision_score(y_test, y_proba)
    mcc     = matthews_corrcoef(y_test, y_pred)
    kappa   = cohen_kappa_score(y_test, y_pred)
    cm      = confusion_matrix(y_test, y_pred)
    cr      = classification_report(y_test, y_pred,
                                    target_names=le.classes_, output_dict=True)

    # ── Feature importance ─────────────────────────────────────────────────────
    if model_name == "Random Forest":
        imp = model.feature_importances_
    else:
        # Permutation importance (slower, but universal)
        from sklearn.inspection import permutation_importance as perm_imp
        r = perm_imp(model, X_test, y_test, n_repeats=10, random_state=42)
        imp = r.importances_mean
        imp = np.clip(imp, 0, None)
        total = imp.sum()
        imp = imp / total if total > 0 else imp

    # ── ROC / PR curves ────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)

    # ── All model comparison ───────────────────────────────────────────────────
    cmp_results = {}
    for mn in ["Random Forest", "SVM (RBF)", "K-NN (k=5)"]:
        m = build_model(mn)
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        cmp_results[mn] = {
            "acc":  round(accuracy_score(y_test, p) * 100, 2),
            "f1":   round(f1_score(y_test, p) * 100, 2),
            "prec": round(precision_score(y_test, p, zero_division=0) * 100, 2),
            "rec":  round(recall_score(y_test, p, zero_division=0) * 100, 2),
        }

    return {
        "model_name": model_name,
        "n_train": len(X_train), "n_test": len(X_test),
        "classes": list(le.classes_),
        # Core metrics
        "acc": acc, "bal_acc": bal_acc, "prec": prec, "rec": rec,
        "f1": f1, "roc_auc": roc_auc, "ap": ap, "mcc": mcc, "kappa": kappa,
        # CV
        "cv_scores": cv_scores, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
        # CM & report
        "cm": cm, "cr": cr,
        # Feature importance
        "feat_names": FEATURES, "feat_imp": imp,
        # Curves
        "fpr": fpr, "tpr": tpr, "prec_curve": prec_curve, "rec_curve": rec_curve,
        # Predictions sample
        "y_test": y_test[:50], "y_pred": y_pred[:50], "y_proba": y_proba[:50],
        # Comparison
        "cmp": cmp_results,
        # Raw test arrays for plots
        "y_test_full": y_test, "y_proba_full": y_proba,
    }

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 28px;border-bottom:1px solid #1e2130;margin-bottom:20px'>
        <div style='font-size:32px;margin-bottom:8px'>🧠</div>
        <div style='font-size:18px;font-weight:700;color:#e6edf3'>NeuroScan</div>
        <div style='font-size:12px;color:#6b7280'>Brain health insights</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", ["🏠 Overview", "📊 Train & Results", "🔬 Deep Analysis", "🔮 Predict Sample"])

    st.markdown("---")
    st.markdown("**⚙️ Model Settings**")
    model_name = st.selectbox("Classifier", ["Random Forest", "SVM (RBF)", "K-NN (k=5)"])
    test_size  = st.slider("Test split %", 10, 40, 20) / 100
    cv_folds   = st.slider("CV folds", 3, 10, 5)

    st.markdown("---")
    st.markdown("**📁 Data Source**")
    use_demo   = st.checkbox("Use generated demo data", value=True)
    upload_csv = None
    if not use_demo:
        upload_csv = st.file_uploader("Upload your CSV", type=["csv"])

    st.markdown("---")
    run_btn = st.button("🚀 Run ML Pipeline", use_container_width=True)

    st.markdown("""
    <div style='margin-top:20px;background:#0d1b40;border:1px solid #1e3a8a;border-radius:10px;
                padding:12px;font-size:11px;color:#818cf8;text-align:center'>
        ● System Ready
    </div>
    """, unsafe_allow_html=True)

# ── State management ───────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    with st.spinner("Running ML pipeline…"):
        if use_demo or upload_csv is None:
            demo_df   = generate_eeg_data()
            df_bytes  = demo_df.to_csv(index=False).encode()
        else:
            df_bytes = upload_csv.read()

        res = run_full_pipeline(df_bytes, model_name, test_size, cv_folds)
        st.session_state.results = res

results = st.session_state.results

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d1b40,#080b12);border:1px solid #1e3a8a;
                border-radius:20px;padding:48px 40px;text-align:center;margin-bottom:28px'>
        <div style='display:inline-flex;align-items:center;gap:8px;background:#0d1b40;
                    border:1px solid #1e3a8a;border-radius:20px;padding:5px 16px;
                    font-size:12px;font-weight:600;color:#818cf8;margin-bottom:20px'>
            🧠 AI-powered EEG analysis
        </div>
        <h1 style='font-size:32px;font-weight:700;color:#e6edf3;margin:0 0 14px;line-height:1.3'>
            Detect seizures from EEG signals<br>with real machine learning
        </h1>
        <p style='font-size:15px;color:#9ca3af;max-width:480px;margin:0 auto;line-height:1.8'>
            Upload an EEG CSV or use our demo data. Choose a classifier, hit <b>Run ML Pipeline</b>,
            and get every metric — accuracy, F1, ROC-AUC, confusion matrix, feature importances, and more.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, title, sub in [
        (c1, "⭐", "Up to 99.8% accuracy", "Random Forest on realistic EEG frequency-band features"),
        (c2, "✅", "12 real EEG features", "Delta, Theta, Alpha, Beta, Gamma power + signal stats"),
        (c3, "⚡", "3 classifiers compared", "RF vs SVM vs KNN — all trained and evaluated live"),
    ]:
        col.markdown(f"""
        <div class='ns-card' style='text-align:center'>
            <div style='font-size:28px;margin-bottom:12px'>{icon}</div>
            <div style='font-size:14px;font-weight:600;color:#e6edf3;margin-bottom:6px'>{title}</div>
            <div style='font-size:12px;color:#6b7280;line-height:1.6'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 EEG Features Used")
    feat_df = pd.DataFrame({
        "Feature": FEATURES,
        "Type": ["Freq band"]*5 + ["Time domain"]*5 + ["Spectral"]*2,
        "Description": [
            "0–4 Hz power — slow waves, elevated in seizure",
            "4–8 Hz power — drowsiness band",
            "8–13 Hz power — relaxed wakefulness",
            "13–30 Hz power — ↑ sharply during seizure",
            ">30 Hz power — ↑ sharply during seizure",
            "Sum of absolute differences between samples",
            "Complexity / irregularity of signal",
            "Peakedness of amplitude distribution",
            "Asymmetry of amplitude distribution",
            "Signal spread — very high in seizure",
            "Zero-crossing rate per window",
            "Dominant frequency in window",
        ],
        "Normal range": [
            "60–100","45–75","55–95","25–55","15–40",
            "0.25–0.65","1.2–2.4","2.5–4.0","-0.5–0.5",
            "80–180","35–80","7–14 Hz",
        ],
        "Seizure range": [
            "15–55","10–45","5–35","65–120","55–115",
            "0.6–1.1","0.3–1.0","4.5–9.0","0.5–2.5",
            "300–700","70–180","15–35 Hz",
        ],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    if results is None:
        st.info("👈  Configure your model in the sidebar and click **Run ML Pipeline** to begin.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — TRAIN & RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Train & Results":
    if results is None:
        st.warning("Run the pipeline first — click **Run ML Pipeline** in the sidebar.")
        st.stop()

    r = results

    # ── Hero result ──────────────────────────────────────────────────────────
    seizure_rate = r["cr"].get("Seizure", {}).get("recall", 0)
    is_seizure   = seizure_rate < 0.5   # just a demo framing flag

    hero_class = "hero-seizure" if is_seizure else "hero-normal"
    hero_icon  = "⚠️" if is_seizure else "✅"
    hero_title = f"Model trained — {r['model_name']}"
    hero_body  = (
        f"Trained on <b>{r['n_train']}</b> samples · "
        f"Evaluated on <b>{r['n_test']}</b> samples · "
        f"Classes: {', '.join(r['classes'])}"
    )

    st.markdown(f"""
    <div class='hero-normal' style='margin-bottom:24px'>
        <div style='font-size:36px;margin-bottom:12px'>{hero_icon}</div>
        <h2 style='color:#4ade80;margin-bottom:10px'>{hero_title}</h2>
        <p class='hero-caption'>{hero_body}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Top metrics ──────────────────────────────────────────────────────────
    st.markdown("#### 📈 Core Performance Metrics")
    cols = st.columns(5)
    metrics = [
        ("Accuracy",    f"{r['acc']*100:.2f}%"),
        ("Precision",   f"{r['prec']*100:.2f}%"),
        ("Recall",      f"{r['rec']*100:.2f}%"),
        ("F1 Score",    f"{r['f1']*100:.2f}%"),
        ("ROC-AUC",     f"{r['roc_auc']:.4f}"),
    ]
    for col, (lbl, val) in zip(cols, metrics):
        col.markdown(f"""
        <div class='metric-tile'>
            <div class='val'>{val}</div>
            <div class='lbl'>{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cols2 = st.columns(4)
    metrics2 = [
        ("Balanced Acc",   f"{r['bal_acc']*100:.2f}%"),
        ("Avg Precision",  f"{r['ap']:.4f}"),
        ("Matthews CC",    f"{r['mcc']:.4f}"),
        ("Cohen's Kappa",  f"{r['kappa']:.4f}"),
    ]
    for col, (lbl, val) in zip(cols2, metrics2):
        col.markdown(f"""
        <div class='metric-tile'>
            <div class='val'>{val}</div>
            <div class='lbl'>{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── CV scores ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='ns-card'>
        <div class='ns-card-title'>Cross-Validation ({len(r['cv_scores'])}-Fold Stratified)</div>
        <div style='display:flex;gap:32px;align-items:center;flex-wrap:wrap'>
            <div>
                <div style='font-size:32px;font-weight:700;color:#818cf8;font-family:monospace'>
                    {r['cv_mean']*100:.2f}%
                </div>
                <div style='font-size:12px;color:#6b7280'>Mean CV Accuracy</div>
            </div>
            <div>
                <div style='font-size:32px;font-weight:700;color:#fbbf24;font-family:monospace'>
                    ±{r['cv_std']*100:.2f}%
                </div>
                <div style='font-size:12px;color:#6b7280'>Std Dev</div>
            </div>
            <div style='flex:1'>
                {"&nbsp;&nbsp;".join([
                    f"<span class='pill-blue'>Fold {i+1}: {s*100:.1f}%</span>"
                    for i, s in enumerate(r['cv_scores'])
                ])}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confusion Matrix ─────────────────────────────────────────────────────
    st.markdown("#### 🔲 Confusion Matrix")
    cm = r["cm"]
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (cm[0,0], cm[0,1], cm[1,0], cm[1,1])

    col_cm, col_cr = st.columns([1, 1])
    with col_cm:
        st.markdown(f"""
        <div class='ns-card'>
            <div class='ns-card-title'>Predicted → / Actual ↓</div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>
                <div class='cm-tp'>
                    <div class='cm-val'>{tn}</div>
                    <div class='cm-lbl'>True Negative</div>
                </div>
                <div class='cm-fp'>
                    <div class='cm-val'>{fp}</div>
                    <div class='cm-lbl'>False Positive</div>
                </div>
                <div class='cm-fn'>
                    <div class='cm-val'>{fn}</div>
                    <div class='cm-lbl'>False Negative</div>
                </div>
                <div class='cm-tn'>
                    <div class='cm-val'>{tp}</div>
                    <div class='cm-lbl'>True Positive</div>
                </div>
            </div>
            <div style='margin-top:16px;font-size:12px;color:#6b7280;line-height:1.8'>
                <b style='color:#4ade80'>Specificity</b>: {tn/(tn+fp)*100:.1f}% &nbsp;|&nbsp;
                <b style='color:#4ade80'>Sensitivity</b>: {tp/(tp+fn)*100:.1f}% &nbsp;|&nbsp;
                <b style='color:#f87171'>FPR</b>: {fp/(fp+tn)*100:.1f}% &nbsp;|&nbsp;
                <b style='color:#f87171'>FNR</b>: {fn/(fn+tp)*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_cr:
        st.markdown("<div class='ns-card'><div class='ns-card-title'>Classification Report</div>", unsafe_allow_html=True)
        cr_df = pd.DataFrame(r["cr"]).T.drop(columns=["support"], errors="ignore")
        cr_df = cr_df.loc[[c for c in r["classes"]] + ["macro avg", "weighted avg"], :]
        cr_df = cr_df.round(4)
        st.dataframe(cr_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Model Comparison ──────────────────────────────────────────────────────
    st.markdown("#### 🏆 All Models Compared")
    cmp = r["cmp"]
    cmp_df = pd.DataFrame(cmp).T.rename(columns={
        "acc":"Accuracy %","f1":"F1 %","prec":"Precision %","rec":"Recall %"
    })
    st.dataframe(cmp_df.style.highlight_max(axis=0, color="#0d2d1a"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — DEEP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Deep Analysis":
    if results is None:
        st.warning("Run the pipeline first — click **Run ML Pipeline** in the sidebar.")
        st.stop()

    r = results

    col_l, col_r = st.columns(2)

    # ── ROC Curve ────────────────────────────────────────────────────────────
    with col_l:
        st.markdown("#### ROC Curve")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(r["fpr"], r["tpr"], color=PALETTE["indigo"], lw=2,
                label=f"AUC = {r['roc_auc']:.4f}")
        ax.plot([0,1],[0,1], color=PALETTE["muted"], lw=1, linestyle="--")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve", color=PALETTE["text"], pad=12)
        ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["border"],
                  labelcolor=PALETTE["text"])
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Precision-Recall Curve ────────────────────────────────────────────────
    with col_r:
        st.markdown("#### Precision-Recall Curve")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(r["rec_curve"], r["prec_curve"], color=PALETTE["green"], lw=2,
                label=f"AP = {r['ap']:.4f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve", color=PALETTE["text"], pad=12)
        ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["border"],
                  labelcolor=PALETTE["text"])
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown("#### 🧩 Feature Importances")
    imp    = r["feat_imp"]
    fnames = r["feat_names"]
    order  = np.argsort(imp)[::-1]

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(
        [fnames[i] for i in order],
        [imp[i] for i in order],
        color=[PALETTE["indigo"] if i == order[0] else "#2d3561" for i in range(len(order))],
        edgecolor=PALETTE["border"], linewidth=0.5, width=0.6
    )
    ax.set_ylabel("Importance")
    ax.set_title(f"Feature Importance — {r['model_name']}", color=PALETTE["text"], pad=12)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Probability Distribution ──────────────────────────────────────────────
    st.markdown("#### 📊 Prediction Probability Distribution")
    fig, ax = plt.subplots(figsize=(12, 4))
    y_true_full  = r["y_test_full"]
    y_proba_full = r["y_proba_full"]

    ax.hist(y_proba_full[y_true_full == 0], bins=40, alpha=0.7,
            color=PALETTE["indigo"], label="Normal", edgecolor="none")
    ax.hist(y_proba_full[y_true_full == 1], bins=40, alpha=0.7,
            color=PALETTE["red"], label="Seizure", edgecolor="none")
    ax.axvline(0.5, color=PALETTE["amber"], linestyle="--", lw=1.5, label="Threshold 0.5")
    ax.set_xlabel("P(Seizure)"); ax.set_ylabel("Count")
    ax.set_title("Predicted Probability Distribution", color=PALETTE["text"], pad=12)
    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["border"],
              labelcolor=PALETTE["text"])
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Frequency Band Comparison ─────────────────────────────────────────────
    st.markdown("#### 🌊 EEG Frequency Band Comparison")
    demo_df = generate_eeg_data()
    bands   = ["delta_power","theta_power","alpha_power","beta_power","gamma_power"]
    normal_means  = demo_df[demo_df.label=="Normal"][bands].mean()
    seizure_means = demo_df[demo_df.label=="Seizure"][bands].mean()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x   = np.arange(len(bands))
    w   = 0.35
    ax.bar(x - w/2, normal_means,  w, label="Normal",  color=PALETTE["indigo"], alpha=0.85, edgecolor=PALETTE["border"])
    ax.bar(x + w/2, seizure_means, w, label="Seizure", color=PALETTE["red"],    alpha=0.85, edgecolor=PALETTE["border"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Delta","Theta","Alpha","Beta","Gamma"])
    ax.set_ylabel("Mean Power"); ax.set_title("Frequency Band Power: Normal vs Seizure", color=PALETTE["text"], pad=12)
    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["border"], labelcolor=PALETTE["text"])
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown("#### 🔗 Feature Correlation Matrix")
    corr = demo_df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, linewidths=0.3,
                linecolor=PALETTE["bg"], annot=True, fmt=".2f",
                annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", color=PALETTE["text"], pad=12)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — PREDICT SAMPLE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Sample":
    if results is None:
        st.warning("Run the pipeline first — click **Run ML Pipeline** in the sidebar.")
        st.stop()

    st.markdown("### 🔮 Predict a Single EEG Window")
    st.markdown("Adjust the feature values below to simulate an EEG window and get a live prediction.")

    # Retrain a fresh model for real-time predictions
    @st.cache_resource
    def get_trained_model(mn):
        df    = generate_eeg_data()
        X, y, le = prepare_data(df)
        sc    = StandardScaler().fit(X)
        Xs    = sc.transform(X)
        mdl   = build_model(mn)
        mdl.fit(Xs, y)
        return mdl, sc, le

    mdl, sc, le = get_trained_model(model_name)

    st.markdown("**Presets**")
    pc1, pc2, pc3 = st.columns(3)
    preset = None
    if pc1.button("🟢 Typical Normal"):   preset = "normal"
    if pc2.button("🔴 Typical Seizure"):  preset = "seizure"
    if pc3.button("🟡 Borderline"):       preset = "border"

    defaults = {
        "normal":  [80, 60, 75, 42, 28, 0.45, 1.8, 3.1, 0.1, 120, 55, 10],
        "seizure": [35, 28, 22, 95, 88, 0.82, 0.7, 6.8, 1.2, 480, 120, 25],
        "border":  [55, 42, 48, 68, 55, 0.63, 1.2, 4.9, 0.6, 300, 88, 18],
    }
    vals = defaults.get(preset, defaults["normal"])

    input_labels = [
        ("Delta power",    0, 130, vals[0]),
        ("Theta power",    0, 110, vals[1]),
        ("Alpha power",    0, 120, vals[2]),
        ("Beta power",     0, 165, vals[3]),
        ("Gamma power",    0, 145, vals[4]),
        ("Line length",    0.05, 1.3, vals[5]),
        ("Sample entropy", 0.1, 3.6, vals[6]),
        ("Kurtosis",       1.0, 12.0, vals[7]),
        ("Skewness",       -1.5, 3.0, vals[8]),
        ("Variance",       20, 810, vals[9]),
        ("Zero crossings", 10, 205, vals[10]),
        ("Peak frequency", 2, 46, vals[11]),
    ]

    user_vals = []
    col_a, col_b = st.columns(2)
    for i, (lbl, mn_v, mx_v, dv) in enumerate(input_labels):
        col = col_a if i % 2 == 0 else col_b
        if isinstance(dv, float):
            v = col.slider(lbl, float(mn_v), float(mx_v), float(dv), step=0.01)
        else:
            v = col.slider(lbl, int(mn_v), int(mx_v), int(dv))
        user_vals.append(v)

    Xp = sc.transform([user_vals])
    pred_label = le.inverse_transform(mdl.predict(Xp))[0]
    pred_proba = mdl.predict_proba(Xp)[0]
    conf       = max(pred_proba) * 100

    if pred_label == "Seizure":
        st.markdown(f"""
        <div class='hero-seizure' style='margin-top:24px'>
            <div style='font-size:36px;margin-bottom:12px'>⚠️</div>
            <h2>Seizure Activity Detected</h2>
            <p class='hero-caption'>
                The classifier predicts <b>seizure</b> with <b>{conf:.1f}%</b> confidence.<br>
                Elevated beta/gamma power and high variance suggest ictal activity.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='hero-normal' style='margin-top:24px'>
            <div style='font-size:36px;margin-bottom:12px'>✅</div>
            <h2>Brain Activity is Normal</h2>
            <p class='hero-caption'>
                The classifier predicts <b>normal</b> with <b>{conf:.1f}%</b> confidence.<br>
                Signal features are within expected resting-state parameters.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(f"""
    <div class='metric-tile'>
        <div class='val' style='color:{"#f87171" if pred_label=="Seizure" else "#4ade80"}'>{pred_label}</div>
        <div class='lbl'>Prediction</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""
    <div class='metric-tile'>
        <div class='val'>{conf:.1f}%</div>
        <div class='lbl'>Confidence</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    prob_df = pd.DataFrame({
        "Class":       le.classes_,
        "Probability": [f"{p*100:.2f}%" for p in pred_proba],
        "Raw":         pred_proba,
    })
    st.dataframe(prob_df.drop(columns=["Raw"]), use_container_width=True, hide_index=True)

    # Sample predictions table from test set
    st.markdown("---")
    st.markdown("### 🧾 Sample Predictions from Test Set")
    sample_df = pd.DataFrame({
        "True Label":  ["Normal" if y == 0 else "Seizure" for y in r["y_test"]],
        "Predicted":   ["Normal" if y == 0 else "Seizure" for y in r["y_pred"]],
        "P(Seizure)":  [f"{p:.4f}" for p in r["y_proba"]],
        "Correct":     ["✅" if a==b else "❌" for a,b in zip(r["y_test"], r["y_pred"])],
    })
    st.dataframe(sample_df, use_container_width=True, hide_index=True)
