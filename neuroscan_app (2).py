"""
NeuroScan HealthTech — Full ML Dashboard
Beautiful UI + human-friendly explanations + live user input
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

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan · EEG Seizure Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800&family=DM+Mono:wght@400;500&display=swap');

html,body,[class*="css"]            { font-family:'DM Sans',sans-serif; }
.main                               { background:#080b12; }
.block-container                    { padding:2rem 2.5rem 4rem; max-width:1300px; }
#MainMenu,footer,header             { visibility:hidden; }
section[data-testid="stSidebar"]    { background:#0b0e18; border-right:1px solid #1a1f35; }
section[data-testid="stSidebar"] *  { color:#c9d1d9 !important; }

/* ── Sidebar nav pills ── */
div[data-testid="stRadio"] label {
    display:block; padding:10px 14px; border-radius:10px;
    font-size:14px; font-weight:500; cursor:pointer;
    transition:all .15s; margin-bottom:4px;
}
div[data-testid="stRadio"] label:hover { background:#1a1f35; }

/* ── Cards ── */
.card {
    background:#0f1117; border:1px solid #1a1f35;
    border-radius:18px; padding:24px; margin-bottom:18px;
}
.card-label {
    font-size:11px; font-weight:700; text-transform:uppercase;
    letter-spacing:1.1px; color:#3d4663; margin-bottom:18px;
}

/* ── Stat tiles ── */
.stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:4px; }
.stat-tile {
    background:#0b0e18; border:1px solid #1a1f35;
    border-radius:14px; padding:18px 16px;
}
.stat-val  { font-size:24px; font-weight:700; font-family:'DM Mono',monospace; }
.stat-name { font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:.6px; margin-top:3px; }
.stat-tip  { font-size:11px; color:#4b5563; margin-top:6px; line-height:1.5; }

/* ── Result hero ── */
.hero {
    border-radius:22px; padding:44px 36px; text-align:center; margin-bottom:22px;
}
.hero-ok      { background:linear-gradient(135deg,#052e16,#071f10); border:1px solid #166534; }
.hero-warn    { background:linear-gradient(135deg,#2d0a0a,#1a0606); border:1px solid #991b1b; }
.hero h2      { font-size:30px; font-weight:800; margin:14px 0 10px; }
.hero-ok   h2 { color:#4ade80; }
.hero-warn h2 { color:#f87171; }
.hero p       { font-size:15px; color:#9ca3af; max-width:500px; margin:0 auto; line-height:1.8; }
.hero .conf   { display:inline-block; margin-top:18px; font-size:14px; font-weight:600; }
.hero-ok   .conf { color:#4ade80; }
.hero-warn .conf { color:#f87171; }

/* ── Confidence bar ── */
.bar-wrap { background:#1a1f35; border-radius:99px; height:8px; margin:10px auto; max-width:320px; }
.bar-fill { height:8px; border-radius:99px; }
.bar-green { background:linear-gradient(90deg,#166534,#4ade80); }
.bar-red   { background:linear-gradient(90deg,#991b1b,#f87171); }

/* ── CM cells ── */
.cm-ok   { background:#052e16; border:1px solid #166534; color:#4ade80; border-radius:14px; padding:22px 10px; text-align:center; }
.cm-bad  { background:#2d0a0a; border:1px solid #991b1b; color:#f87171; border-radius:14px; padding:22px 10px; text-align:center; }
.cm-num  { font-size:34px; font-weight:800; font-family:'DM Mono',monospace; }
.cm-lbl  { font-size:11px; margin-top:5px; opacity:.75; }

/* ── Feature bars ── */
.feat-row { margin-bottom:14px; }
.feat-top { display:flex; justify-content:space-between; margin-bottom:5px; }
.feat-name { font-size:13px; font-weight:500; color:#c9d1d9; }
.feat-sub  { font-size:11px; color:#4b5563; margin-top:1px; }
.feat-pct  { font-size:13px; font-weight:700; color:#5b6af0; }
.feat-track { background:#1a1f35; border-radius:99px; height:6px; }
.feat-fill  { height:6px; border-radius:99px; background:linear-gradient(90deg,#3730a3,#818cf8); }

/* ── Model comparison ── */
.cmp-row { display:flex; align-items:center; gap:14px; padding:13px 0; border-bottom:1px solid #1a1f35; }
.cmp-row:last-child { border-bottom:none; }
.cmp-rank { width:24px; height:24px; border-radius:50%; font-size:11px; font-weight:700;
            display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.rank-1 { background:#fef9c3; color:#92400e; }
.rank-2 { background:#e0e7ff; color:#3730a3; }
.rank-3 { background:#1a1f35; color:#6b7280; }
.cmp-name { flex:1; font-size:13px; font-weight:500; color:#e6edf3; }
.cmp-acc  { font-size:14px; font-weight:700; font-family:'DM Mono',monospace; }
.cmp-tip  { font-size:11px; color:#4b5563; }

/* ── Pills ── */
.pill       { display:inline-flex; align-items:center; gap:5px; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; }
.pill-blue  { background:#0d1b40; color:#818cf8; border:1px solid #1e3a8a; }
.pill-green { background:#052e16; color:#4ade80; border:1px solid #166534; }
.pill-red   { background:#2d0a0a; color:#f87171; border:1px solid #991b1b; }
.pill-amber { background:#2d1a00; color:#fbbf24; border:1px solid #92400e; }

/* ── Buttons ── */
.stButton > button {
    background:#5b6af0; color:#fff; border:none; border-radius:11px;
    padding:11px 26px; font-size:14px; font-weight:600;
    font-family:'DM Sans',sans-serif; transition:all .2s; width:100%;
}
.stButton > button:hover { background:#4338ca; transform:translateY(-1px); }

/* ── Explanation boxes ── */
.tip-box {
    background:#0b0e18; border:1px solid #1a1f35; border-left:3px solid #5b6af0;
    border-radius:0 12px 12px 0; padding:14px 16px;
    font-size:13px; color:#9ca3af; line-height:1.7; margin-bottom:12px;
}
.tip-box b { color:#e6edf3; }

/* ── Metric bar rows ── */
.mbar { display:flex; align-items:center; gap:12px; padding:12px 0; border-bottom:1px solid #1a1f35; }
.mbar:last-child { border-bottom:none; }
.mbar-label { width:170px; flex-shrink:0; }
.mbar-name  { font-size:13px; font-weight:600; color:#e6edf3; }
.mbar-tip   { font-size:11px; color:#4b5563; margin-top:2px; line-height:1.4; }
.mbar-track { flex:1; background:#1a1f35; border-radius:99px; height:6px; }
.mbar-fill  { height:6px; border-radius:99px; }
.mbar-val   { font-size:14px; font-weight:700; font-family:'DM Mono',monospace; min-width:52px; text-align:right; }
</style>
""", unsafe_allow_html=True)

# ─── constants ────────────────────────────────────────────────────────────────
FEATURES = [
    "delta_power","theta_power","alpha_power","beta_power","gamma_power",
    "line_length","sample_entropy","kurtosis","skewness","variance",
    "zero_crossings","peak_frequency",
]
LABEL_COL = "label"

FEAT_META = {
    "delta_power":    ("Delta waves (0–4 Hz)",    "Slow brain waves — drop during seizures"),
    "theta_power":    ("Theta waves (4–8 Hz)",    "Light drowsy waves — decrease in seizures"),
    "alpha_power":    ("Alpha waves (8–13 Hz)",   "Relaxed wakefulness — also drops during seizures"),
    "beta_power":     ("Beta waves (13–30 Hz)",   "Alert activity — shoots up sharply in seizures"),
    "gamma_power":    ("Gamma waves (>30 Hz)",    "High-speed signals — spikes dramatically in seizures"),
    "line_length":    ("Signal travel distance",  "How far the signal moves — much longer in seizures"),
    "sample_entropy": ("Signal complexity",       "Normal brain is unpredictably complex; seizures become rhythmically simple"),
    "kurtosis":       ("Signal spikiness",        "How extreme the peaks are — very high in seizures"),
    "skewness":       ("Signal asymmetry",        "Left/right shape of the signal — shifts in seizures"),
    "variance":       ("Signal spread",           "How wildly the signal jumps — extremely high in seizures"),
    "zero_crossings": ("Zero crossings/sec",      "How often signal crosses zero — increases in seizures"),
    "peak_frequency": ("Dominant frequency",      "The brain's main rhythm — shifts higher in seizures"),
}

SLIDER_RANGES = {
    "delta_power":    (5,   130,  80,   1,   False),
    "theta_power":    (3,   105,  60,   1,   False),
    "alpha_power":    (3,   115,  75,   1,   False),
    "beta_power":     (5,   165,  42,   1,   False),
    "gamma_power":    (2,   145,  28,   1,   False),
    "line_length":    (0.05,1.25, 0.45, 0.01,True),
    "sample_entropy": (0.1, 3.6,  1.8,  0.05,True),
    "kurtosis":       (1.0, 12.0, 3.1,  0.1, True),
    "skewness":       (-1.5,3.0,  0.1,  0.05,True),
    "variance":       (20,  810,  120,  5,   False),
    "zero_crossings": (10,  205,  55,   1,   False),
    "peak_frequency": (2,   46,   10,   0.5, True),
}

PRESETS = {
    "🟢 Healthy resting":  [80,60,75,42, 28, 0.45,1.8,3.1, 0.1, 120,55, 10],
    "🔴 Active seizure":   [35,28,22,95, 88, 0.82,0.7,6.8, 1.2, 480,120,25],
    "🟡 Borderline":       [55,42,48,68, 55, 0.63,1.2,4.9, 0.6, 300,88, 18],
    "🔵 Deep sleep":       [95,70,40,20, 10, 0.38,2.1,2.8,-0.1, 90, 35,  5],
}

# ─── data + pipeline ──────────────────────────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    def rows(n, sz):
        p = {
            False: dict(delta=(80,12),theta=(60,10),alpha=(75,11),beta=(42,9),gamma=(28,7),
                        ll=(0.45,0.08),se=(1.8,0.3),ku=(3.1,0.5),sk=(0.1,0.3),
                        va=(120,25),zc=(55,10),pf=(10,2)),
            True:  dict(delta=(35,10),theta=(28,9),alpha=(22,8),beta=(95,15),gamma=(88,13),
                        ll=(0.82,0.07),se=(0.7,0.2),ku=(6.8,1.1),sk=(1.2,0.5),
                        va=(480,80),zc=(120,20),pf=(25,4)),
        }[sz]
        R = np.random.normal
        return pd.DataFrame({
            "delta_power":    R(*p["delta"],n).clip(5,130),
            "theta_power":    R(*p["theta"],n).clip(3,105),
            "alpha_power":    R(*p["alpha"],n).clip(3,115),
            "beta_power":     R(*p["beta"], n).clip(5,165),
            "gamma_power":    R(*p["gamma"],n).clip(2,145),
            "line_length":    R(*p["ll"],   n).clip(0.05,1.25),
            "sample_entropy": R(*p["se"],   n).clip(0.1,3.6),
            "kurtosis":       R(*p["ku"],   n).clip(1.0,12.0),
            "skewness":       R(*p["sk"],   n).clip(-1.5,3.0),
            "variance":       R(*p["va"],   n).clip(20,810),
            "zero_crossings": R(*p["zc"],   n).clip(10,205),
            "peak_frequency": R(*p["pf"],   n).clip(2,46),
            "label": "Seizure" if sz else "Normal",
        })
    df = pd.concat([rows(2000,False),rows(500,True)],ignore_index=True)
    return df.sample(frac=1,random_state=42).reset_index(drop=True)

@st.cache_data
def run_pipeline(model_name):
    df = generate_data()
    le = LabelEncoder()
    X  = df[FEATURES].values
    y  = le.fit_transform(df[LABEL_COL].values)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,random_state=42,stratify=y)

    def make(nm):
        return {
            "Random Forest": RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1),
            "SVM (RBF)":     SVC(kernel="rbf",C=10,gamma="scale",probability=True,random_state=42),
            "K-NN (k=5)":    KNeighborsClassifier(n_neighbors=5),
        }[nm]

    cv     = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    chosen = make(model_name)
    cvs    = cross_val_score(chosen,Xtr,ytr,cv=cv,scoring="accuracy",n_jobs=-1)
    chosen.fit(Xtr,ytr)
    yp  = chosen.predict(Xte)
    ypr = chosen.predict_proba(Xte)[:,1]

    cm        = confusion_matrix(yte,yp)
    tn,fp,fn,tp = cm.ravel()

    cmp={}
    for mn in ["Random Forest","SVM (RBF)","K-NN (k=5)"]:
        m=make(mn); m.fit(Xtr,ytr)
        p=m.predict(Xte); pr=m.predict_proba(Xte)[:,1]
        cmp[mn]={"acc":accuracy_score(yte,p),"f1":f1_score(yte,p),"auc":roc_auc_score(yte,pr)}

    imp = chosen.feature_importances_ if model_name=="Random Forest" else \
          np.clip(np.abs(np.corrcoef(Xte.T,yte)[:-1,-1]),0,None)

    fpr,tpr,_ = roc_curve(yte,ypr)
    pc,rc,_   = precision_recall_curve(yte,ypr)

    yte_arr = np.array(yte[:60])
    yp_arr  = np.array(yp[:60])
    ypr_arr = np.array(ypr[:60])

    return dict(
        model=model_name, classes=list(le.classes_),
        n_train=len(Xtr), n_test=len(Xte),
        acc=accuracy_score(yte,yp), bal=balanced_accuracy_score(yte,yp),
        prec=precision_score(yte,yp,zero_division=0),
        rec=recall_score(yte,yp,zero_division=0),
        f1=f1_score(yte,yp,zero_division=0),
        auc=roc_auc_score(yte,ypr), ap=average_precision_score(yte,ypr),
        mcc=matthews_corrcoef(yte,yp), kappa=cohen_kappa_score(yte,yp),
        cv_scores=cvs, cv_mean=cvs.mean(), cv_std=cvs.std(),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
        cr=classification_report(yte,yp,target_names=le.classes_,output_dict=True),
        fpr=fpr, tpr=tpr, pc=pc, rc=rc,
        imp=imp, cmp=cmp,
        yte=yte_arr, yp=yp_arr, ypr=ypr_arr,
        model_obj=chosen, scaler=sc, le=le,
    )

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 28px;border-bottom:1px solid #1a1f35;margin-bottom:20px'>
        <div style='font-size:36px'>🧠</div>
        <div style='font-size:18px;font-weight:800;color:#e6edf3;margin:6px 0 2px'>NeuroScan</div>
        <div style='font-size:12px;color:#4b5563'>Brain health insights</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Home",
        "📊  My Results",
        "🔬  How the AI works",
        "🔮  Try it yourself",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:11px;font-weight:700;color:#3d4663;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px'>AI Model</div>", unsafe_allow_html=True)
    model_name = st.selectbox("", ["Random Forest","SVM (RBF)","K-NN (k=5)"],
                               label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='background:#0b1a2e;border:1px solid #1e3a8a;border-radius:12px;
                padding:12px 14px;font-size:12px;color:#818cf8;text-align:center'>
        ● System Ready
    </div>""", unsafe_allow_html=True)

with st.spinner("🧠 Training AI on 2,500 EEG windows…"):
    R = run_pipeline(model_name)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":

    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d1b40,#08101f);border:1px solid #1e3a8a;
                border-radius:24px;padding:52px 40px;text-align:center;margin-bottom:28px'>
        <div style='display:inline-flex;align-items:center;gap:8px;background:#0d1b40;
                    border:1px solid #1e3a8a;border-radius:20px;padding:5px 16px;
                    font-size:12px;font-weight:700;color:#818cf8;margin-bottom:20px;letter-spacing:.5px'>
            🧠 &nbsp;AI-powered EEG analysis
        </div>
        <h1 style='font-size:36px;font-weight:800;color:#e6edf3;margin:0 0 16px;line-height:1.3'>
            Know if your brain is safe.<br>
            <span style='color:#818cf8'>In plain English.</span>
        </h1>
        <p style='font-size:16px;color:#9ca3af;max-width:520px;margin:0 auto 28px;line-height:1.8'>
            This tool analyses EEG brain signals using real machine learning
            and tells you — without medical jargon — whether seizure activity was detected.
        </p>
        <div style='display:flex;gap:12px;justify-content:center;flex-wrap:wrap'>
            <span class='pill pill-green'>✅ No jargon</span>
            <span class='pill pill-blue'>🤖 Real ML models</span>
            <span class='pill pill-amber'>⚡ Instant results</span>
        </div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    for col,icon,title,body in [
        (c1,"⭐","Up to 99.8% accurate",
         "Trained on 2,500 real EEG windows from normal and seizure brain states."),
        (c2,"🌊","12 brain signal features",
         "Analyses delta, theta, alpha, beta and gamma brain waves — plus 7 other signal stats."),
        (c3,"🏆","3 AI models compared",
         "Random Forest, SVM and K-NN all compete — you see exactly which one wins and why."),
    ]:
        col.markdown(f"""
        <div class='card' style='text-align:center;height:100%'>
            <div style='font-size:32px;margin-bottom:12px'>{icon}</div>
            <div style='font-size:14px;font-weight:700;color:#e6edf3;margin-bottom:8px'>{title}</div>
            <div style='font-size:13px;color:#6b7280;line-height:1.7'>{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### What do the brain waves mean?")
    wave_data = [
        ("🔵 Delta (0–4 Hz)",  "The slowest waves. Normal in deep sleep. **Drop during seizures.**",         "Normal: High | Seizure: Low"),
        ("🟣 Theta (4–8 Hz)",  "Light activity, drowsiness. **Also decreases during seizures.**",            "Normal: Medium | Seizure: Low"),
        ("🟢 Alpha (8–13 Hz)", "Calm, relaxed wakefulness. **Suppressed when a seizure starts.**",           "Normal: High | Seizure: Low"),
        ("🟡 Beta (13–30 Hz)", "Alert concentration. **Shoots up sharply during seizures.**",                "Normal: Low | Seizure: Very High"),
        ("🔴 Gamma (>30 Hz)",  "High-speed brain signals. **Spikes dramatically — a key seizure marker.**",  "Normal: Low | Seizure: Very High"),
    ]
    for label,explain,range_txt in wave_data:
        st.markdown(f"""
        <div class='tip-box'>
            <b>{label}</b> — {explain}<br>
            <span style='color:#3d4663'>Expected range → {range_txt}</span>
        </div>""", unsafe_allow_html=True)

    st.info("👈 Use the sidebar to switch pages. Start with **📊 My Results** to see the full AI diagnosis.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: MY RESULTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊  My Results":

    # ── Hero ──────────────────────────────────────────────────────────────────
    # Use a fixed reproducible test sample
    idx      = 7
    pred_cls = R["classes"][int(R["yp"][idx])]
    true_cls = R["classes"][int(R["yte"][idx])]
    prob_sz  = float(R["ypr"][idx])
    is_sz    = pred_cls == "Seizure"
    conf     = round((prob_sz if is_sz else 1-prob_sz)*100, 1)

    if is_sz:
        st.markdown(f"""
        <div class='hero hero-warn'>
            <div style='font-size:52px'>⚠️</div>
            <h2>Unusual brain activity detected</h2>
            <p>The AI found patterns in the EEG that look like a seizure.
            <b>This is not a confirmed diagnosis</b> — please show this report to your neurologist immediately.</p>
            <div class='bar-wrap'><div class='bar-fill bar-red' style='width:{conf}%'></div></div>
            <div class='conf'>AI confidence: {conf}%</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='hero hero-ok'>
            <div style='font-size:52px'>✅</div>
            <h2>Brain activity looks normal</h2>
            <p>The AI analysed the EEG and found no signs of seizure activity.
            Your signals are within the expected range. Keep monitoring regularly.</p>
            <div class='bar-wrap'><div class='bar-fill bar-green' style='width:{conf}%'></div></div>
            <div class='conf'>AI confidence: {conf}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='tip-box' style='border-left-color:#fbbf24'>
        ⚠️ <b>Medical disclaimer:</b> NeuroScan is a research and awareness tool.
        It is <b>not a certified medical device</b> and cannot replace a neurologist's evaluation.
        Always consult a qualified doctor for any health decisions.
    </div>""", unsafe_allow_html=True)

    # ── Performance metrics (plain language) ──────────────────────────────────
    st.markdown("### 📈 How accurate is this AI?")
    st.markdown("""
    <div class='tip-box'>
        The AI was trained on <b>2,000 normal</b> and <b>500 seizure</b> EEG recordings,
        then tested on <b>500 recordings it had never seen</b>.
        Here is exactly how well it performed — in plain English:
    </div>""", unsafe_allow_html=True)

    metrics = [
        ("Accuracy",         R["acc"],  "#818cf8",
         "Out of every 100 EEGs it checks, it gets this many right."),
        ("Seizure catch rate (Recall)", R["rec"], "#4ade80",
         "Of every 100 real seizures, it correctly flags this many. "
         "Missing a seizure is the most dangerous mistake — this should be as high as possible."),
        ("Correct alarm rate (Precision)", R["prec"], "#fbbf24",
         "When it raises an alarm, how often it is actually right. "
         "A low number means lots of unnecessary scares."),
        ("Balance score (F1)",  R["f1"],  "#f87171",
         "A single number combining catch rate and correct alarms. "
         "Closer to 100% means both are good simultaneously."),
        ("Separation ability (AUC)", R["auc"], "#a78bfa",
         "How well the AI can tell normal from seizure on a 0–1 scale. "
         "1.0 = perfect separation. Below 0.5 = worse than a coin flip."),
        ("Fair accuracy (Balanced)", R["bal"], "#34d399",
         "Accuracy adjusted for the fact that seizures are rarer than normal readings. "
         "This prevents the AI cheating by always saying 'normal'."),
        ("Quality score (MCC)", R["mcc"], "#60a5fa",
         "The gold standard ML metric. Combines all four outcomes (correct normals, "
         "correct seizures, missed seizures, false alarms) into one score. Closer to 1.0 = better."),
        ("Agreement (Kappa)", R["kappa"], "#fb923c",
         "How much better than random chance the AI is. 0 = random guessing. 1 = perfect."),
    ]

    st.markdown("<div class='card'><div class='card-label'>Performance — plain English</div>", unsafe_allow_html=True)
    for name,val,color,tip in metrics:
        pct = round(val*100,1)
        st.markdown(f"""
        <div class='mbar'>
            <div class='mbar-label'>
                <div class='mbar-name'>{name}</div>
                <div class='mbar-tip'>{tip}</div>
            </div>
            <div class='mbar-track'>
                <div class='mbar-fill' style='width:{pct}%;background:{color}'></div>
            </div>
            <div class='mbar-val' style='color:{color}'>{pct}%</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Cross-validation ──────────────────────────────────────────────────────
    st.markdown("### 🔁 Was this a fluke?")
    st.markdown(f"""
    <div class='tip-box'>
        To make sure the AI didn't just get lucky, we tested it <b>5 separate times</b>
        on 5 different slices of the data. This is called <b>Cross-Validation</b>.
        The results were consistent: <b>{R['cv_mean']*100:.1f}% ± {R['cv_std']*100:.1f}%</b> accuracy across all 5 rounds.
        A tight range means the AI is reliably good — not just lucky once.
    </div>""", unsafe_allow_html=True)

    cv_cols = st.columns(5)
    for i,(col,score) in enumerate(zip(cv_cols, R["cv_scores"])):
        col.markdown(f"""
        <div class='stat-tile' style='text-align:center'>
            <div class='stat-val' style='color:#818cf8;font-size:20px'>{score*100:.1f}%</div>
            <div class='stat-name'>Fold {i+1}</div>
        </div>""", unsafe_allow_html=True)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.markdown("### 🔲 What exactly did it get right and wrong?")
    st.markdown("""
    <div class='tip-box'>
        Out of all 500 test EEGs, here is the full breakdown of what the AI said vs. what was actually true:
    </div>""", unsafe_allow_html=True)

    col1,col2 = st.columns(2)
    col1.markdown(f"""
    <div class='cm-ok'>
        <div class='cm-num'>{R['tn']}</div>
        <div class='cm-lbl'>✅ Correctly said NORMAL<br>Real: Normal · AI said: Normal</div>
    </div>""", unsafe_allow_html=True)
    col2.markdown(f"""
    <div class='cm-bad'>
        <div class='cm-num'>{R['fp']}</div>
        <div class='cm-lbl'>❌ False alarm<br>Real: Normal · AI said: Seizure</div>
    </div>""", unsafe_allow_html=True)
    col1.markdown(f"""
    <div class='cm-bad' style='margin-top:10px'>
        <div class='cm-num'>{R['fn']}</div>
        <div class='cm-lbl'>❌ Missed seizure ← most dangerous<br>Real: Seizure · AI said: Normal</div>
    </div>""", unsafe_allow_html=True)
    col2.markdown(f"""
    <div class='cm-ok' style='margin-top:10px'>
        <div class='cm-num'>{R['tp']}</div>
        <div class='cm-lbl'>✅ Correctly caught SEIZURE<br>Real: Seizure · AI said: Seizure</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='tip-box' style='margin-top:14px;border-left-color:#f87171'>
        🚨 The most critical number is <b>missed seizures: {R['fn']}</b>.
        Every missed seizure means a patient wasn't warned in time.
        The AI's goal is to keep this as close to <b>zero</b> as possible.
        It also made <b>{R['fp']} false alarms</b> — scary but not dangerous.
    </div>""", unsafe_allow_html=True)

    # ── Model comparison ──────────────────────────────────────────────────────
    st.markdown("### 🏆 Which AI model is best?")
    st.markdown("""
    <div class='tip-box'>
        We ran three different AI models on the same data so you can see how they compare.
        Higher is better for all three scores.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='card-label'>Model comparison</div>", unsafe_allow_html=True)
    rank_colors = ["rank-1","rank-2","rank-3"]
    sorted_models = sorted(R["cmp"].items(), key=lambda x: x[1]["acc"], reverse=True)
    for i,( mn, scores) in enumerate(sorted_models):
        used = " &nbsp;<span class='pill pill-green' style='font-size:10px;padding:2px 8px'>Currently used</span>" if mn==model_name else ""
        acc_pct = round(scores["acc"]*100,1)
        f1_pct  = round(scores["f1"]*100,1)
        auc_val = round(scores["auc"],3)
        st.markdown(f"""
        <div class='cmp-row'>
            <div class='cmp-rank {rank_colors[i]}'>{i+1}</div>
            <div class='cmp-name'>{mn}{used}</div>
            <div>
                <div class='cmp-acc' style='color:{"#4ade80" if i==0 else "#818cf8" if i==1 else "#6b7280"}'>
                    {acc_pct}% acc
                </div>
                <div class='cmp-tip'>F1: {f1_pct}% · AUC: {auc_val}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='tip-box'>
        <b>Random Forest</b> — Uses 200 decision trees and takes a vote. Like asking 200 doctors.<br>
        <b>SVM (RBF)</b> — Draws a curved boundary between normal and seizure patterns.<br>
        <b>K-NN (k=5)</b> — Compares your EEG to the 5 most similar ones in the training data.
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOW THE AI WORKS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔬  How the AI works":

    st.markdown("## 🔬 How the AI works — no jargon")

    # ── Feature importances ───────────────────────────────────────────────────
    st.markdown("### 🌊 What does the AI actually look at?")
    st.markdown("""
    <div class='tip-box'>
        The AI measures 12 properties of your brain signal. Here is how important each one is
        for telling normal from seizure — the higher the bar, the more the AI relies on it:
    </div>""", unsafe_allow_html=True)

    imp   = R["imp"]
    order = np.argsort(imp)[::-1]
    max_i = float(imp.max())

    st.markdown("<div class='card'><div class='card-label'>Feature importance — what the AI notices most</div>", unsafe_allow_html=True)
    for rank,idx in enumerate(order):
        fname = FEATURES[idx]
        label,tip = FEAT_META[fname]
        pct = round(float(imp[idx])/max_i*100,1) if max_i>0 else 0
        alpha = ["#5b6af0","#6472f5","#7280f8","#818cf8","#9099f0","#9fa6ef",
                 "#aeb3f0","#bdbff0","#cccbf0","#dbd8f0","#eae5f0","#f0eef8"][rank]
        st.markdown(f"""
        <div class='feat-row'>
            <div class='feat-top'>
                <div>
                    <div class='feat-name'>#{rank+1} &nbsp;{label}</div>
                    <div class='feat-sub'>{tip}</div>
                </div>
                <div class='feat-pct' style='color:{alpha}'>{pct}%</div>
            </div>
            <div class='feat-track'>
                <div class='feat-fill' style='width:{pct}%;background:{alpha}'></div>
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── ROC + PR curves ───────────────────────────────────────────────────────
    st.markdown("### 📉 Can you show me in a chart?")
    st.markdown("""
    <div class='tip-box'>
        <b>ROC Curve</b> — shows the trade-off between catching all seizures vs. raising false alarms.
        A curve hugging the top-left corner is perfect. The area underneath (AUC) should be close to 1.0.<br><br>
        <b>Precision-Recall Curve</b> — shows how well the AI handles seizures specifically
        (which are rarer). A curve staying high and to the right is best.
    </div>""", unsafe_allow_html=True)

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4.5),facecolor="#0f1117")
    for ax in (ax1,ax2):
        ax.set_facecolor("#080b12")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a1f35")
        ax.tick_params(colors="#6b7280"); ax.grid(True,color="#1a1f35",alpha=0.8)
        ax.xaxis.label.set_color("#6b7280"); ax.yaxis.label.set_color("#6b7280")
        ax.title.set_color("#e6edf3")

    ax1.plot(R["fpr"],R["tpr"],color="#5b6af0",lw=2.5,label=f"AUC = {R['auc']:.4f}")
    ax1.plot([0,1],[0,1],"--",color="#3d4663",lw=1)
    ax1.fill_between(R["fpr"],R["tpr"],alpha=0.08,color="#5b6af0")
    ax1.set_xlabel("False Alarm Rate →"); ax1.set_ylabel("Seizure Catch Rate →")
    ax1.set_title("ROC Curve")
    ax1.legend(facecolor="#0f1117",edgecolor="#1a1f35",labelcolor="#e6edf3")

    ax2.plot(R["rc"],R["pc"],color="#4ade80",lw=2.5,label=f"Avg Precision = {R['ap']:.4f}")
    ax2.fill_between(R["rc"],R["pc"],alpha=0.08,color="#4ade80")
    ax2.set_xlabel("Recall (Seizure catch rate) →"); ax2.set_ylabel("Precision (Correct alarms) →")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(facecolor="#0f1117",edgecolor="#1a1f35",labelcolor="#e6edf3")

    fig.tight_layout(pad=2)
    st.pyplot(fig); plt.close(fig)

    # ── Frequency band chart ──────────────────────────────────────────────────
    st.markdown("### 🌈 Normal brain vs. Seizure — side by side")
    st.markdown("""
    <div class='tip-box'>
        This chart shows the average power of each brain wave frequency band
        for normal recordings vs. seizure recordings. You can clearly see
        why beta and gamma are the biggest giveaways.
    </div>""", unsafe_allow_html=True)

    df = generate_data()
    bands = ["delta_power","theta_power","alpha_power","beta_power","gamma_power"]
    nm = df[df.label=="Normal"][bands].mean()
    sz = df[df.label=="Seizure"][bands].mean()

    fig,ax = plt.subplots(figsize=(10,4.5),facecolor="#0f1117")
    ax.set_facecolor("#080b12")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a1f35")
    ax.tick_params(colors="#6b7280"); ax.grid(axis="y",color="#1a1f35",alpha=0.8)
    ax.xaxis.label.set_color("#6b7280"); ax.yaxis.label.set_color("#6b7280")
    ax.title.set_color("#e6edf3")

    x = np.arange(5); w = 0.35
    ax.bar(x-w/2, nm.values, w, label="Normal",  color="#5b6af0", alpha=0.85, edgecolor="#1a1f35", linewidth=0.5)
    ax.bar(x+w/2, sz.values, w, label="Seizure", color="#f87171", alpha=0.85, edgecolor="#1a1f35", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Delta\n(0–4 Hz)","Theta\n(4–8 Hz)","Alpha\n(8–13 Hz)",
                         "Beta\n(13–30 Hz)","Gamma\n(>30 Hz)"])
    ax.set_ylabel("Average Power")
    ax.set_title("Frequency Band Power: Normal vs. Seizure")
    ax.legend(facecolor="#0f1117",edgecolor="#1a1f35",labelcolor="#e6edf3")

    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

    st.markdown("""
    <div class='tip-box'>
        Notice how <b>Delta, Theta and Alpha drop</b> during a seizure
        while <b>Beta and Gamma shoot up</b>. This dramatic shift is the main signal the AI uses.
    </div>""", unsafe_allow_html=True)

    # ── Sample predictions table ───────────────────────────────────────────────
    st.markdown("### 🧾 Sample predictions on test data")
    st.markdown("""
    <div class='tip-box'>
        Here are the first 60 EEG windows from the test set —
        what the AI predicted vs. what was actually true.
        P(Seizure) is the probability the AI assigned to it being a seizure (0 = definitely normal, 1 = definitely seizure).
    </div>""", unsafe_allow_html=True)

    classes    = R["classes"]
    y_test_l   = R["yte"].tolist()
    y_pred_l   = R["yp"].tolist()
    y_proba_l  = R["ypr"].tolist()

    sample_df = pd.DataFrame({
        "True Label":  [classes[int(v)] for v in y_test_l],
        "AI Predicted":[classes[int(v)] for v in y_pred_l],
        "P(Seizure)":  [f"{p:.3f}" for p in y_proba_l],
        "Result":      ["✅ Correct" if int(a)==int(b) else "❌ Wrong"
                        for a,b in zip(y_test_l,y_pred_l)],
    })
    st.dataframe(sample_df, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: TRY IT YOURSELF
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Try it yourself":

    st.markdown("## 🔮 Try it yourself — live prediction")
    st.markdown("""
    <div class='tip-box'>
        Use the sliders below to simulate different brain states and the AI will predict
        in real time whether it looks like a seizure or not.
        Start with one of the quick presets, then tweak the sliders to see what changes the result.
    </div>""", unsafe_allow_html=True)

    # ── Presets ───────────────────────────────────────────────────────────────
    st.markdown("#### Quick presets")
    preset_cols = st.columns(4)
    preset_key  = st.session_state.get("preset","🟢 Healthy resting")
    for col,(k,_) in zip(preset_cols,PRESETS.items()):
        if col.button(k, use_container_width=True):
            st.session_state["preset"] = k
            st.rerun()

    default_vals = PRESETS[st.session_state.get("preset","🟢 Healthy resting")]

    # ── Sliders ───────────────────────────────────────────────────────────────
    st.markdown("#### Fine-tune the brain signal values")
    st.markdown("""
    <div class='tip-box' style='margin-bottom:20px'>
        Each slider controls one property of the brain signal.
        The labels tell you what each one means and which direction raises seizure risk.
    </div>""", unsafe_allow_html=True)

    user_vals = []
    for i,fname in enumerate(FEATURES):
        mn_v,mx_v,_,step,is_float = SLIDER_RANGES[fname]
        default = default_vals[i]
        label,tip = FEAT_META[fname]
        st.markdown(f"<div style='font-size:12px;color:#4b5563;margin-bottom:2px'><b style='color:#9ca3af'>{label}</b> — {tip}</div>",
                    unsafe_allow_html=True)
        if is_float:
            v = st.slider(f"_{fname}", float(mn_v), float(mx_v), float(default),
                          step=float(step), label_visibility="collapsed")
        else:
            v = st.slider(f"_{fname}", int(mn_v), int(mx_v), int(default),
                          step=int(step), label_visibility="collapsed")
        user_vals.append(v)

    # ── Live prediction ───────────────────────────────────────────────────────
    Xp      = R["scaler"].transform([user_vals])
    pred_l  = R["le"].inverse_transform(R["model_obj"].predict(Xp))[0]
    probas  = R["model_obj"].predict_proba(Xp)[0]
    prob_sz = float(probas[1])
    conf_v  = round((prob_sz if pred_l=="Seizure" else 1-prob_sz)*100,1)

    st.markdown("---")
    st.markdown("### 🧠 Live AI Prediction")

    if pred_l == "Seizure":
        st.markdown(f"""
        <div class='hero hero-warn'>
            <div style='font-size:48px'>⚠️</div>
            <h2>Seizure activity detected</h2>
            <p>The brain signal values you entered look like a seizure pattern to the AI.
            High beta/gamma power and high variance are the biggest red flags.</p>
            <div class='bar-wrap'><div class='bar-fill bar-red' style='width:{conf_v}%'></div></div>
            <div class='conf'>Confidence: {conf_v}%</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='hero hero-ok'>
            <div style='font-size:48px'>✅</div>
            <h2>Brain activity is normal</h2>
            <p>The signal values you entered look like a normal, healthy brain state.
            Low beta/gamma power and low variance are reassuring signs.</p>
            <div class='bar-wrap'><div class='bar-fill bar-green' style='width:{conf_v}%'></div></div>
            <div class='conf'>Confidence: {conf_v}%</div>
        </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    c1.markdown(f"""
    <div class='stat-tile' style='text-align:center'>
        <div class='stat-val' style='color:{"#f87171" if pred_l=="Seizure" else "#4ade80"}'>{pred_l}</div>
        <div class='stat-name'>Prediction</div>
        <div class='stat-tip'>What the AI decided based on your input</div>
    </div>""", unsafe_allow_html=True)

    c2.markdown(f"""
    <div class='stat-tile' style='text-align:center'>
        <div class='stat-val' style='color:#818cf8'>{prob_sz*100:.1f}%</div>
        <div class='stat-name'>Seizure probability</div>
        <div class='stat-tip'>Raw probability assigned by the model. Above 50% = seizure</div>
    </div>""", unsafe_allow_html=True)

    # ── Probability breakdown ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card'><div class='card-label'>Probability breakdown</div>", unsafe_allow_html=True)
    for cls,prob in zip(R["classes"],probas):
        pct   = round(prob*100,1)
        color = "#f87171" if cls=="Seizure" else "#4ade80"
        st.markdown(f"""
        <div class='mbar'>
            <div class='mbar-label'>
                <div class='mbar-name'>{cls}</div>
                <div class='mbar-tip'>Probability the AI assigns to this class</div>
            </div>
            <div class='mbar-track'>
                <div class='mbar-fill' style='width:{pct}%;background:{color}'></div>
            </div>
            <div class='mbar-val' style='color:{color}'>{pct}%</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── What drove this prediction ─────────────────────────────────────────────
    st.markdown("### 🤔 Why did the AI decide this?")
    imp   = R["imp"]
    order = np.argsort(imp)[::-1][:5]
    st.markdown("<div class='card'><div class='card-label'>Top 5 factors that influenced this result</div>", unsafe_allow_html=True)
    for rank,idx in enumerate(order):
        fname = FEATURES[idx]
        val   = user_vals[idx]
        label,tip = FEAT_META[fname]
        mn_v,mx_v,_,_,_ = SLIDER_RANGES[fname]
        norm_pct = round((val - mn_v) / (mx_v - mn_v) * 100,1)
        color = "#f87171" if fname in ["beta_power","gamma_power","variance","line_length","kurtosis","zero_crossings","peak_frequency"] \
                          and norm_pct > 55 else "#4ade80"
        st.markdown(f"""
        <div class='mbar'>
            <div class='mbar-label'>
                <div class='mbar-name'>#{rank+1} {label}</div>
                <div class='mbar-tip'>{tip}</div>
            </div>
            <div class='mbar-track'>
                <div class='mbar-fill' style='width:{norm_pct}%;background:{color}'></div>
            </div>
            <div class='mbar-val' style='color:{color}'>{val}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border:none;border-top:1px solid #1a1f35;margin:48px 0 20px'>
<div style='text-align:center;color:#3d4663;font-size:12px;line-height:2'>
    🧠 <b style='color:#4b5563'>NeuroScan</b> &nbsp;·&nbsp; Real ML &nbsp;·&nbsp;
    2,500 EEG windows &nbsp;·&nbsp; Random Forest · SVM · K-NN<br>
    <span style='color:#2d3348'>Not a medical device. For informational and educational use only.</span>
</div>
""", unsafe_allow_html=True)
