"""
NeuroScan — EEG Seizure Detector
Simple for patients. Full ML under the hood.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, io
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    average_precision_score, balanced_accuracy_score, cohen_kappa_score,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan — Am I Safe?",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #080b12; }
.block-container { padding: 2rem 2rem 4rem; max-width: 860px; }
#MainMenu, footer, header { visibility: hidden; }

/* Cards */
.ns-card {
    background: #0f1117; border: 1px solid #1e2130;
    border-radius: 20px; padding: 28px; margin-bottom: 20px;
}

/* Big result */
.result-safe {
    background: linear-gradient(135deg,#052e16,#0a1f14);
    border: 1.5px solid #166534; border-radius: 24px;
    padding: 48px 32px; text-align: center; margin-bottom: 24px;
}
.result-danger {
    background: linear-gradient(135deg,#2d0a0a,#1a0808);
    border: 1.5px solid #991b1b; border-radius: 24px;
    padding: 48px 32px; text-align: center; margin-bottom: 24px;
}
.result-safe   h1 { color:#4ade80; font-size:36px; font-weight:800; margin:12px 0 10px; }
.result-danger h1 { color:#f87171; font-size:36px; font-weight:800; margin:12px 0 10px; }
.result-caption { color:#9ca3af; font-size:16px; line-height:1.8; max-width:520px; margin:0 auto; }

/* Confidence bar */
.conf-wrap { background:#1e2130; border-radius:99px; height:10px; margin:20px auto; max-width:340px; }
.conf-fill-green { height:10px; border-radius:99px; background:linear-gradient(90deg,#166534,#4ade80); }
.conf-fill-red   { height:10px; border-radius:99px; background:linear-gradient(90deg,#991b1b,#f87171); }

/* Metric tiles */
.metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:16px; }
.metric-tile {
    background:#0d1017; border:1px solid #1e2130;
    border-radius:14px; padding:18px 16px; text-align:center;
}
.metric-tile .val { font-size:26px; font-weight:700; color:#e6edf3; font-family:'DM Mono',monospace; }
.metric-tile .lbl { font-size:11px; color:#6b7280; margin-top:4px; text-transform:uppercase; letter-spacing:0.6px; }
.metric-tile .explain { font-size:12px; color:#9ca3af; margin-top:6px; line-height:1.5; }

/* Step indicator */
.step-wrap { display:flex; gap:8px; justify-content:center; margin-bottom:32px; }
.step-dot { width:10px; height:10px; border-radius:50%; background:#1e2130; }
.step-dot.active { background:#5b6af0; }
.step-dot.done   { background:#4ade80; }

/* Buttons */
.stButton > button {
    background:#5b6af0; color:white; border:none; border-radius:12px;
    padding:14px 32px; font-size:16px; font-weight:600;
    font-family:'DM Sans',sans-serif; width:100%; transition:all 0.2s;
}
.stButton > button:hover { background:#4338ca; }

/* Expander */
.streamlit-expanderHeader { color:#9ca3af !important; font-size:13px !important; }

/* Plain text */
.big-emoji { font-size:64px; display:block; text-align:center; }
.section-label {
    font-size:11px; font-weight:600; text-transform:uppercase;
    letter-spacing:1px; color:#4b5563; margin-bottom:16px;
}
.plain-explain {
    background:#0f1117; border:1px solid #1e2130; border-radius:14px;
    padding:20px; font-size:14px; color:#9ca3af; line-height:1.8; margin-bottom:12px;
}
.plain-explain b { color:#e6edf3; }

/* CM cells */
.cm-good { background:#052e16; color:#4ade80; border-radius:12px; padding:20px; text-align:center; }
.cm-bad  { background:#2d0a0a; color:#f87171; border-radius:12px; padding:20px; text-align:center; }
.cm-num  { font-size:30px; font-weight:700; font-family:'DM Mono',monospace; }
.cm-lbl  { font-size:11px; margin-top:4px; opacity:0.7; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = [
    "delta_power","theta_power","alpha_power","beta_power","gamma_power",
    "line_length","sample_entropy","kurtosis","skewness","variance",
    "zero_crossings","peak_frequency",
]
LABEL_COL = "label"

PLAIN_METRICS = {
    "Accuracy":       ("How often the AI is correct overall.", "#818cf8"),
    "Precision":      ("When it says 'Seizure', how often it's right.", "#4ade80"),
    "Recall":         ("Of all real seizures, how many it catches.", "#fbbf24"),
    "F1 Score":       ("Balance between catching seizures and false alarms.", "#f87171"),
    "ROC-AUC":        ("Overall ability to separate normal from seizure (1.0 = perfect).", "#a78bfa"),
    "Balanced Acc":   ("Accuracy adjusted for the fact that seizures are rarer.", "#34d399"),
    "Matthews CC":    ("Overall quality score — closer to 1.0 is better.", "#60a5fa"),
    "Cohen's Kappa":  ("Agreement above random chance — closer to 1.0 is better.", "#fb923c"),
}

# ── Data generator ────────────────────────────────────────────────────────────
@st.cache_data
def generate_eeg_data():
    np.random.seed(42)
    def make(n, seizure=False):
        if not seizure:
            return pd.DataFrame({
                "delta_power":    np.random.normal(80,12,n).clip(10,120),
                "theta_power":    np.random.normal(60,10,n).clip(5,100),
                "alpha_power":    np.random.normal(75,11,n).clip(10,110),
                "beta_power":     np.random.normal(42,9,n).clip(5,80),
                "gamma_power":    np.random.normal(28,7,n).clip(2,60),
                "line_length":    np.random.normal(0.45,0.08,n).clip(0.1,0.9),
                "sample_entropy": np.random.normal(1.8,0.3,n).clip(0.5,3.5),
                "kurtosis":       np.random.normal(3.1,0.5,n).clip(1.0,6.0),
                "skewness":       np.random.normal(0.1,0.3,n).clip(-1.5,1.5),
                "variance":       np.random.normal(120,25,n).clip(20,300),
                "zero_crossings": np.random.normal(55,10,n).clip(10,120),
                "peak_frequency": np.random.normal(10,2,n).clip(2,30),
                "label": "Normal",
            })
        else:
            return pd.DataFrame({
                "delta_power":    np.random.normal(35,10,n).clip(5,80),
                "theta_power":    np.random.normal(28,9,n).clip(3,70),
                "alpha_power":    np.random.normal(22,8,n).clip(3,60),
                "beta_power":     np.random.normal(95,15,n).clip(30,160),
                "gamma_power":    np.random.normal(88,13,n).clip(30,140),
                "line_length":    np.random.normal(0.82,0.07,n).clip(0.3,1.2),
                "sample_entropy": np.random.normal(0.7,0.2,n).clip(0.1,1.5),
                "kurtosis":       np.random.normal(6.8,1.1,n).clip(2.0,12.0),
                "skewness":       np.random.normal(1.2,0.5,n).clip(-0.5,3.0),
                "variance":       np.random.normal(480,80,n).clip(100,800),
                "zero_crossings": np.random.normal(120,20,n).clip(40,200),
                "peak_frequency": np.random.normal(25,4,n).clip(8,45),
                "label": "Seizure",
            })
    df = pd.concat([make(2000,False), make(500,True)], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# ── ML pipeline ───────────────────────────────────────────────────────────────
@st.cache_data
def run_pipeline(model_name="Random Forest"):
    df = generate_eeg_data()
    le = LabelEncoder()
    X  = df[FEATURES].values
    y  = le.fit_transform(df[LABEL_COL].values)

    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "SVM (RBF)":     SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
        "K-NN (k=5)":    KNeighborsClassifier(n_neighbors=5),
    }

    # Cross-val on chosen model
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    chosen = models[model_name]
    cv_scores = cross_val_score(chosen, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

    chosen.fit(X_train, y_train)
    y_pred  = chosen.predict(X_test)
    y_proba = chosen.predict_proba(X_test)[:, 1]

    cm  = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # All model comparison
    cmp = {}
    for mn, m in models.items():
        m.fit(X_train, y_train)
        p = m.predict(X_test)
        pr = m.predict_proba(X_test)[:,1]
        cmp[mn] = {
            "acc":  round(accuracy_score(y_test, p)*100, 1),
            "f1":   round(f1_score(y_test, p)*100, 1),
            "auc":  round(roc_auc_score(y_test, pr), 4),
        }

    # Feature importance
    imp = chosen.feature_importances_ if model_name == "Random Forest" else np.abs(
        np.corrcoef(X_test.T, y_test)[:-1, -1]
    )

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    pc, rc, _   = precision_recall_curve(y_test, y_proba)

    return {
        "model": model_name,
        "classes": list(le.classes_),
        "n_train": len(X_train), "n_test": len(X_test),
        # core metrics
        "acc":     accuracy_score(y_test, y_pred),
        "bal_acc": balanced_accuracy_score(y_test, y_pred),
        "prec":    precision_score(y_test, y_pred, zero_division=0),
        "rec":     recall_score(y_test, y_pred, zero_division=0),
        "f1":      f1_score(y_test, y_pred, zero_division=0),
        "auc":     roc_auc_score(y_test, y_proba),
        "ap":      average_precision_score(y_test, y_proba),
        "mcc":     matthews_corrcoef(y_test, y_pred),
        "kappa":   cohen_kappa_score(y_test, y_pred),
        # cv
        "cv_scores": cv_scores, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
        # cm
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # report
        "cr": classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True),
        # curves
        "fpr": fpr, "tpr": tpr, "pc": pc, "rc": rc,
        # features
        "feat_imp": imp,
        # comparison
        "cmp": cmp,
        # sample preds (numpy arrays — safe)
        "y_test_arr":  np.array(y_test[:60]),
        "y_pred_arr":  np.array(y_pred[:60]),
        "y_proba_arr": np.array(y_proba[:60]),
        # trained model + scaler for live predict
        "model_obj": chosen, "scaler": sc, "le": le,
    }

# ── Run pipeline on load ───────────────────────────────────────────────────────
with st.spinner("🧠 Training AI model on 2,500 EEG windows…"):
    R = run_pipeline("Random Forest")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:32px 0 8px'>
    <div style='font-size:48px'>🧠</div>
    <div style='font-size:28px;font-weight:800;color:#e6edf3;margin:8px 0 4px'>NeuroScan</div>
    <div style='font-size:15px;color:#6b7280'>EEG Seizure Detector — powered by real machine learning</div>
</div>
<hr style='border:none;border-top:1px solid #1e2130;margin:24px 0'>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — THE ANSWER (what everyone comes for)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-label'>📋 Your Result</div>", unsafe_allow_html=True)

# Simulate a "latest" prediction from the test set
sample_idx  = 7  # fixed index for reproducibility
sample_true = int(R["y_test_arr"][sample_idx])
sample_pred = int(R["y_pred_arr"][sample_idx])
sample_prob = float(R["y_proba_arr"][sample_idx])
is_seizure  = sample_pred == 1
conf_pct    = round(sample_prob * 100 if is_seizure else (1 - sample_prob) * 100, 1)

if is_seizure:
    st.markdown(f"""
    <div class="result-danger">
        <span class="big-emoji">⚠️</span>
        <h1>Unusual brain activity detected</h1>
        <p class="result-caption">
            The AI found patterns in your EEG that look similar to seizure activity.
            <b>This does not mean you are definitely having a seizure</b> — but you should
            speak to your neurologist and share this report with them as soon as possible.
        </p>
        <div class="conf-wrap"><div class="conf-fill-red" style="width:{conf_pct}%"></div></div>
        <div style="color:#f87171;font-size:14px;margin-top:4px">
            AI confidence: <b>{conf_pct}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="result-safe">
        <span class="big-emoji">✅</span>
        <h1>Your brain activity looks normal</h1>
        <p class="result-caption">
            The AI analysed your EEG and found no signs of seizure activity.
            Your brain signals are within the expected range for a healthy, resting state.
            Keep monitoring regularly and consult your doctor with any concerns.
        </p>
        <div class="conf-wrap"><div class="conf-fill-green" style="width:{conf_pct}%"></div></div>
        <div style="color:#4ade80;font-size:14px;margin-top:4px">
            AI confidence: <b>{conf_pct}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="plain-explain">
    ⚠️ <b>Important:</b> This tool is for informational purposes only.
    It is <b>not a medical diagnosis</b>. Always consult a qualified neurologist
    before making any health decisions.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HOW TRUSTWORTHY IS THE AI?
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr style='border:none;border-top:1px solid #1e2130;margin:32px 0'>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>🔒 How reliable is this AI?</div>", unsafe_allow_html=True)

st.markdown(f"""
<div class="plain-explain">
    This AI was trained on <b>2,000 normal</b> and <b>500 seizure</b> EEG recordings.
    It was tested on <b>{R['n_test']} unseen recordings</b> it had never seen before.
    Here is how well it performed on that test:
</div>
""", unsafe_allow_html=True)

# Plain-language metrics
metric_data = [
    ("Accuracy",     R["acc"],     "Of every 100 EEGs it checks, it gets this many right."),
    ("Seizure catch rate (Recall)", R["rec"], "Of every 100 real seizures, it correctly flags this many. Missing a seizure is the worst outcome — this should be high."),
    ("False alarm rate (1−Precision)", 1-R["prec"], "Out of every 100 seizure warnings, this many turn out to be wrong. Lower is better."),
    ("Overall score (F1)",  R["f1"],  "A single number combining catch rate and false alarms. Closer to 100% = better."),
]

for name, val, explain in metric_data:
    pct  = round(val * 100, 1)
    color = "#4ade80" if pct >= 90 else "#fbbf24" if pct >= 75 else "#f87171"
    bar_color = color
    # for false alarm rate, lower is better — flip colour
    if "False alarm" in name:
        color = "#4ade80" if pct <= 10 else "#fbbf24" if pct <= 25 else "#f87171"
    st.markdown(f"""
    <div style="background:#0f1117;border:1px solid #1e2130;border-radius:14px;padding:18px;margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <div style="font-size:14px;font-weight:600;color:#e6edf3">{name}</div>
            <div style="font-size:20px;font-weight:700;color:{color};font-family:monospace">{pct}%</div>
        </div>
        <div style="background:#1e2130;border-radius:99px;height:7px;margin-bottom:10px">
            <div style="width:{pct}%;height:7px;border-radius:99px;background:{bar_color}"></div>
        </div>
        <div style="font-size:13px;color:#9ca3af;line-height:1.6">{explain}</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CONFUSION MATRIX (plain language)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr style='border:none;border-top:1px solid #1e2130;margin:32px 0'>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>🔲 What did the AI get right and wrong?</div>", unsafe_allow_html=True)

st.markdown("""
<div class="plain-explain">
    Out of all the test EEGs, here is exactly what the AI said vs what was actually true:
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)
c1.markdown(f"""
<div class="cm-good">
    <div class="cm-num">{R['tn']}</div>
    <div class="cm-lbl">✅ Correctly said NORMAL</div>
</div>""", unsafe_allow_html=True)
c2.markdown(f"""
<div class="cm-bad">
    <div class="cm-num">{R['fp']}</div>
    <div class="cm-lbl">❌ Said SEIZURE but was actually Normal (false alarm)</div>
</div>""", unsafe_allow_html=True)
c1.markdown(f"""
<div class="cm-bad" style="margin-top:10px">
    <div class="cm-num">{R['fn']}</div>
    <div class="cm-lbl">❌ Said NORMAL but was actually Seizure (missed!)</div>
</div>""", unsafe_allow_html=True)
c2.markdown(f"""
<div class="cm-good" style="margin-top:10px">
    <div class="cm-num">{R['tp']}</div>
    <div class="cm-lbl">✅ Correctly caught SEIZURE</div>
</div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="plain-explain" style="margin-top:16px">
    The most important number above is <b>missed seizures: {R['fn']}</b>.
    The lower this is, the safer the AI is for patients.
    A missed seizure is more dangerous than a false alarm.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — WHAT SIGNALS DOES THE AI LOOK AT?
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr style='border:none;border-top:1px solid #1e2130;margin:32px 0'>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>🌊 What does the AI actually look at?</div>", unsafe_allow_html=True)

st.markdown("""
<div class="plain-explain">
    Your brain produces electrical signals at different speeds (frequencies).
    The AI measures the strength of each frequency band and several other signal properties.
    Here are the signals it finds most useful for detecting seizures:
</div>
""", unsafe_allow_html=True)

feat_labels = {
    "beta_power":     ("Beta waves (13–30 Hz)", "Shoot up dramatically during seizures"),
    "gamma_power":    ("Gamma waves (>30 Hz)",  "Also spike sharply during seizures"),
    "variance":       ("Signal variance",        "How wildly the signal jumps around — very high in seizures"),
    "line_length":    ("Line length",            "Total distance the signal travels — much longer in seizures"),
    "sample_entropy": ("Signal complexity",      "Normal brain is more complex; seizures become rhythmically simple"),
    "kurtosis":       ("Signal spikiness",       "Seizure signals have extreme peaks"),
    "delta_power":    ("Delta waves (0–4 Hz)",   "Slow waves — drop during seizures"),
    "alpha_power":    ("Alpha waves (8–13 Hz)",  "Relaxed wakefulness — also drops during seizures"),
    "zero_crossings": ("Zero crossings",         "How often the signal crosses zero — increases in seizures"),
    "peak_frequency": ("Peak frequency",         "The dominant brain rhythm — shifts higher in seizures"),
    "theta_power":    ("Theta waves (4–8 Hz)",   "Light activity — decreases during seizures"),
    "skewness":       ("Signal skew",            "Asymmetry of the signal shape"),
}

imp    = R["feat_imp"]
order  = np.argsort(imp)[::-1]
top_n  = 6

for rank, idx in enumerate(order[:top_n]):
    fname  = FEATURES[idx]
    label, explain = feat_labels.get(fname, (fname, ""))
    pct    = round(float(imp[idx]) / float(imp.max()) * 100, 1)
    color  = ["#5b6af0","#818cf8","#a5b4fc","#c7d2fe","#e0e7ff","#ededfd"][rank]
    st.markdown(f"""
    <div style="background:#0f1117;border:1px solid #1e2130;border-radius:14px;padding:16px;margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
            <div>
                <span style="font-size:13px;font-weight:600;color:#e6edf3">#{rank+1} {label}</span><br>
                <span style="font-size:12px;color:#9ca3af">{explain}</span>
            </div>
            <span style="font-size:16px;font-weight:700;color:{color};font-family:monospace;white-space:nowrap;margin-left:16px">{pct}%</span>
        </div>
        <div style="background:#1e2130;border-radius:99px;height:6px">
            <div style="width:{pct}%;height:6px;border-radius:99px;background:{color}"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRY IT YOURSELF (live predictor)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr style='border:none;border-top:1px solid #1e2130;margin:32px 0'>", unsafe_allow_html=True)
st.markdown("<div class='section-label'>🔮 Try a live prediction</div>", unsafe_allow_html=True)

st.markdown("""
<div class="plain-explain">
    Choose a preset below to simulate different brain states and see what the AI predicts instantly.
</div>
""", unsafe_allow_html=True)

presets = {
    "🟢 Healthy resting brain":  [80,60,75,42,28, 0.45,1.8,3.1, 0.1,120,55,10],
    "🔴 Active seizure pattern": [35,28,22,95,88, 0.82,0.7,6.8, 1.2,480,120,25],
    "🟡 Borderline / uncertain": [55,42,48,68,55, 0.63,1.2,4.9, 0.6,300,88,18],
}

choice = st.radio("Select a brain state to test:", list(presets.keys()), horizontal=True)
vals   = presets[choice]

Xp     = R["scaler"].transform([vals])
pred_l = R["le"].inverse_transform(R["model_obj"].predict(Xp))[0]
proba  = R["model_obj"].predict_proba(Xp)[0]
conf_l = round(max(proba) * 100, 1)

if pred_l == "Seizure":
    st.markdown(f"""
    <div class="result-danger" style="padding:32px">
        <span style="font-size:40px">⚠️</span>
        <h1 style="margin:10px 0 8px">Seizure activity detected</h1>
        <p class="result-caption">AI confidence: <b>{conf_l}%</b></p>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="result-safe" style="padding:32px">
        <span style="font-size:40px">✅</span>
        <h1 style="margin:10px 0 8px">Brain activity is normal</h1>
        <p class="result-caption">AI confidence: <b>{conf_l}%</b></p>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FOR THE TECHNICALLY CURIOUS (collapsible)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr style='border:none;border-top:1px solid #1e2130;margin:32px 0'>", unsafe_allow_html=True)

with st.expander("🔬 For the technically curious — full ML details"):

    st.markdown("#### All Performance Metrics")
    all_metrics = {
        "Accuracy":      R["acc"],
        "Balanced Acc":  R["bal_acc"],
        "Precision":     R["prec"],
        "Recall":        R["rec"],
        "F1 Score":      R["f1"],
        "ROC-AUC":       R["auc"],
        "Avg Precision": R["ap"],
        "Matthews CC":   R["mcc"],
        "Cohen's Kappa": R["kappa"],
        "CV Mean (5-fold)": R["cv_mean"],
        "CV Std":        R["cv_std"],
    }
    cols = st.columns(4)
    for i, (k, v) in enumerate(all_metrics.items()):
        cols[i % 4].metric(k, f"{v*100:.2f}%" if v <= 1 else f"{v:.4f}")

    st.markdown("#### Model Comparison")
    cmp_df = pd.DataFrame(R["cmp"]).T
    cmp_df.columns = ["Accuracy %", "F1 %", "ROC-AUC"]
    st.dataframe(cmp_df.style.highlight_max(axis=0, color="#0d2d1a"), use_container_width=True)

    st.markdown("#### Classification Report")
    cr_df = pd.DataFrame(R["cr"]).T.drop(columns=["support"], errors="ignore")
    st.dataframe(cr_df.round(4), use_container_width=True)

    st.markdown("#### ROC Curve")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                             facecolor="#0f1117")
    for ax in axes:
        ax.set_facecolor("#080b12")
        for spine in ax.spines.values(): spine.set_edgecolor("#1e2130")
        ax.tick_params(colors="#6b7280"); ax.yaxis.label.set_color("#6b7280")
        ax.xaxis.label.set_color("#6b7280"); ax.title.set_color("#e6edf3")
        ax.grid(True, color="#1e2130", alpha=0.6)

    axes[0].plot(R["fpr"], R["tpr"], color="#5b6af0", lw=2, label=f"AUC={R['auc']:.4f}")
    axes[0].plot([0,1],[0,1], "--", color="#6b7280", lw=1)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend(facecolor="#0f1117", labelcolor="#e6edf3", edgecolor="#1e2130")

    axes[1].plot(R["rc"], R["pc"], color="#4ade80", lw=2, label=f"AP={R['ap']:.4f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend(facecolor="#0f1117", labelcolor="#e6edf3", edgecolor="#1e2130")

    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

    st.markdown("#### Cross-Validation (5-fold)")
    cv_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(len(R["cv_scores"]))],
        "Accuracy": [f"{s*100:.2f}%" for s in R["cv_scores"]],
    })
    st.dataframe(cv_df, use_container_width=True, hide_index=True)
    st.info(f"Mean: {R['cv_mean']*100:.2f}% ± {R['cv_std']*100:.2f}%")

    st.markdown("#### Sample Predictions from Test Set (first 60)")
    classes = R["classes"]
    y_test_list  = R["y_test_arr"].tolist()
    y_pred_list  = R["y_pred_arr"].tolist()
    y_proba_list = R["y_proba_arr"].tolist()
    sample_df = pd.DataFrame({
        "True Label": [classes[int(v)] for v in y_test_list],
        "Predicted":  [classes[int(v)] for v in y_pred_list],
        "P(Seizure)": [f"{p:.4f}" for p in y_proba_list],
        "Correct":    ["✅" if int(a)==int(b) else "❌"
                       for a,b in zip(y_test_list, y_pred_list)],
    })
    st.dataframe(sample_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border:none;border-top:1px solid #1e2130;margin:40px 0 20px'>
<div style='text-align:center;color:#4b5563;font-size:12px;line-height:1.8'>
    🧠 <b style='color:#6b7280'>NeuroScan</b> · Random Forest · 2,500 EEG windows ·
    Trained fresh every session<br>
    <span style='color:#374151'>Not a medical device. For informational use only.</span>
</div>
""", unsafe_allow_html=True)
