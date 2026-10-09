"""DataLens - upload a dataset, audit it, clean it, and benchmark ML models in one place."""
import html as htmllib
import io
import time
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, adjusted_rand_score, calinski_harabasz_score, confusion_matrix,
                             davies_bouldin_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
st.set_page_config(page_title="DataLens", page_icon="🔎", layout="wide")

# ----------------------------------------------------------------------------
# Look & feel
# ----------------------------------------------------------------------------
TEAL, AMBER, CORAL, VIOLET = "#4FA89B", "#C9A24D", "#D17572", "#8F89D1"
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700;800&family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Sora', sans-serif !important; letter-spacing: -0.02em; }
.block-container { padding-top: 2rem; max-width: 1300px; }
.hero { padding: 2.2rem 2.4rem; border-radius: 22px; margin-bottom: 1.4rem;
  background: radial-gradient(900px 300px at 85% -20%, rgba(79,168,155,.16), transparent 60%),
              radial-gradient(700px 300px at 0% 120%, rgba(143,137,209,.14), transparent 60%), #101a2e;
  border: 1px solid rgba(255,255,255,.07); }
.hero h1 { font-size: 3rem; margin: 0; font-weight: 800; }
.hero p { color: #9fb0cc; font-size: 1.1rem; margin: .5rem 0 0 0; max-width: 46rem; }
.score { font-family: 'Sora', sans-serif; font-size: 4rem; font-weight: 800; line-height: 1; }
.scorelabel { color: #9fb0cc; font-size: .95rem; margin-top: .3rem; }
.pill { display:inline-block; padding: .15rem .7rem; border-radius: 999px; font-size: .8rem;
  background: rgba(79,168,155,.12); color: #4FA89B; margin-right: .4rem; }
[data-testid="stMetric"] { background: #131D31; border: 1px solid rgba(255,255,255,.06);
  padding: 1rem 1.2rem; border-radius: 16px; }
[data-testid="stMetricValue"] { font-family: 'Sora', sans-serif; }
.stTabs [data-baseweb="tab-list"] { gap: .4rem; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: .5rem 1rem; background: #131D31; }
.stTabs [aria-selected="true"] { background: rgba(79,168,155,.14) !important; }
.stButton > button { border-radius: 12px; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)


def style(fig, h=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      colorway=[TEAL, AMBER, VIOLET, CORAL, "#6F9FD8", "#9DBF6E"],
                      font=dict(family="DM Sans"))
    return fig


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def load_sample(name):
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
    rng = np.random.default_rng(42)
    if name == "Breast cancer (messy: nulls + duplicates)":
        df = load_breast_cancer(as_frame=True).frame.iloc[:, [0, 1, 2, 3, 4, 5, 30]].copy()
        df.columns = [c.replace(" ", "_") for c in df.columns]
        for c in df.columns[:-1]:
            df.loc[rng.choice(df.index, 40, replace=False), c] = np.nan
        df = pd.concat([df, df.sample(25, random_state=1)], ignore_index=True)
        return df
    if name == "Wine (classification)":
        return load_wine(as_frame=True).frame
    return load_diabetes(as_frame=True).frame  # regression


def read_file(file):
    if file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)


def outlier_counts(df):
    out = {}
    for c in df.select_dtypes("number").columns:
        q1, q3 = df[c].quantile([.25, .75])
        iqr = q3 - q1
        out[c] = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
    return pd.Series(out, dtype=int)


def quality_score(df):
    comp = 1 - df.isna().sum().sum() / max(df.size, 1)
    uniq = 1 - df.duplicated().sum() / max(len(df), 1)
    oc = outlier_counts(df)
    num_cells = max(len(df) * max(len(oc), 1), 1)
    out = 1 - oc.sum() / num_cells
    return round(100 * (0.55 * comp + 0.30 * uniq + 0.15 * out), 1)


def clean_data(df, drop_dups, col_thresh, num_strat, cat_strat):
    d, log = df.copy(), []
    if drop_dups:
        n = int(d.duplicated().sum())
        d = d.drop_duplicates()
        log.append(f"Removed {n} duplicate rows")
    miss = d.isna().mean() * 100
    drop_cols = miss[miss > col_thresh].index.tolist()
    if drop_cols:
        d = d.drop(columns=drop_cols)
        log.append(f"Dropped columns above {col_thresh}% missing: {', '.join(drop_cols)}")
    num = d.select_dtypes("number").columns.tolist()
    cat = [c for c in d.columns if c not in num]
    n_num = int(d[num].isna().sum().sum()) if num else 0
    n_cat = int(d[cat].isna().sum().sum()) if cat else 0
    if num and n_num:
        if num_strat == "Drop rows":
            d = d.dropna(subset=num)
        elif num_strat.startswith("KNN"):
            d[num] = KNNImputer(n_neighbors=5).fit_transform(d[num])
        else:
            d[num] = SimpleImputer(strategy=num_strat.lower()).fit_transform(d[num])
        log.append(f"Filled {n_num} numeric gaps using {num_strat}")
    if cat and n_cat:
        if cat_strat == "Drop rows":
            d = d.dropna(subset=cat)
        elif cat_strat == "Mode":
            for c in cat:
                m = d[c].mode()
                d[c] = d[c].fillna(m.iloc[0] if len(m) else "Unknown")
        else:
            d[cat] = d[cat].fillna("Missing")
        log.append(f"Filled {n_cat} categorical gaps using {cat_strat}")
    return d.reset_index(drop=True), log or ["Nothing to fix. The data was already clean."]


def make_pre(X):
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num and X[c].nunique() <= 30]
    parts = []
    if num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore",
                                                            sparse_output=False))]), cat))
    return ColumnTransformer(parts, remainder="drop")


CLF = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "KNN": lambda: KNeighborsClassifier(),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=42),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=150, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42),
    "SVM": lambda: SVC(),
    "Naive Bayes": lambda: GaussianNB(),
}
REG = {
    "Linear Regression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(),
    "Lasso": lambda: Lasso(alpha=0.01),
    "ElasticNet": lambda: ElasticNet(alpha=0.01),
    "KNN": lambda: KNeighborsRegressor(),
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=42),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=150, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=42),
    "SVR": lambda: SVR(),
}


def run_models(df, target, features, task, test_size, chosen):
    data = df[features + [target]].dropna(subset=[target])
    if len(data) > 20000:
        data = data.sample(20000, random_state=42)
    X, y = data[features], data[target]
    strat = y if task == "Classification" and y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)
    zoo = CLF if task == "Classification" else REG
    rows, fitted = [], {}
    bar = st.progress(0.0, text="Training models...")
    for i, name in enumerate(chosen):
        pipe = Pipeline([("pre", make_pre(X)), ("model", zoo[name]())])
        t0 = time.time()
        pipe.fit(Xtr, ytr)
        secs = time.time() - t0
        pred = pipe.predict(Xte)
        try:
            cv = cross_val_score(pipe, X, y, cv=5,
                                 scoring="accuracy" if task == "Classification" else "r2").mean()
        except Exception:
            cv = np.nan
        if task == "Classification":
            rows.append({"Model": name, "Accuracy": accuracy_score(yte, pred),
                         "Precision": precision_score(yte, pred, average="weighted", zero_division=0),
                         "Recall": recall_score(yte, pred, average="weighted", zero_division=0),
                         "F1": f1_score(yte, pred, average="weighted", zero_division=0),
                         "CV Accuracy (5-fold)": cv, "Train time (s)": secs})
        else:
            rows.append({"Model": name, "R2": r2_score(yte, pred),
                         "MAE": mean_absolute_error(yte, pred),
                         "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
                         "CV R2 (5-fold)": cv, "Train time (s)": secs})
        fitted[name] = (pipe, yte, pred)
        bar.progress((i + 1) / len(chosen), text=f"Trained {name}")
    bar.empty()
    key = "Accuracy" if task == "Classification" else "R2"
    return pd.DataFrame(rows).sort_values(key, ascending=False).reset_index(drop=True), fitted


def importances(pipe):
    try:
        names = pipe.named_steps["pre"].get_feature_names_out()
        m = pipe.named_steps["model"]
        if hasattr(m, "feature_importances_"):
            vals = m.feature_importances_
        elif hasattr(m, "coef_"):
            vals = np.abs(np.atleast_2d(m.coef_)).mean(axis=0)
        else:
            return None
        s = pd.Series(vals, index=[n.split("__", 1)[-1] for n in names])
        return s.sort_values(ascending=False).head(12)[::-1]
    except Exception:
        return None


def cluster_scores(Xs, labels):
    mask = labels != -1
    if mask.sum() < 3 or len(set(labels[mask])) < 2:
        return np.nan, np.nan, np.nan
    n = min(len(Xs[mask]), 5000)
    return (silhouette_score(Xs[mask], labels[mask], sample_size=n, random_state=42),
            davies_bouldin_score(Xs[mask], labels[mask]),
            calinski_harabasz_score(Xs[mask], labels[mask]))


def accuracy_suite(df, target, task, test_size):
    """Score every algorithm: supervised (train/test/CV), KNN sweep, and clustering."""
    feats = [c for c in df.columns if c != target]
    data = df[feats + [target]].dropna(subset=[target])
    if len(data) > 20000:
        data = data.sample(20000, random_state=42)
    X, y = data[feats], data[target]
    clf = task == "Classification"
    strat = y if clf and y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)
    scoring = "accuracy" if clf else "r2"
    rows = []
    for name, make in (CLF if clf else REG).items():
        pipe = Pipeline([("pre", make_pre(X)), ("model", make())]).fit(Xtr, ytr)
        try:
            cv = cross_val_score(pipe, X, y, cv=5, scoring=scoring).mean() * 100
        except Exception:
            cv = np.nan
        tr, te = pipe.score(Xtr, ytr) * 100, pipe.score(Xte, yte) * 100
        rows.append({"Algorithm": name, "Train %": tr, "Test %": te, "CV %": cv,
                     "Overfit gap": tr - te})
    sup = pd.DataFrame(rows).sort_values("Test %", ascending=False).reset_index(drop=True)

    sub = data.sample(min(len(data), 5000), random_state=42)
    knn_rows = []
    for k in range(1, 26, 2):
        est = KNeighborsClassifier(n_neighbors=k) if clf else KNeighborsRegressor(n_neighbors=k)
        pipe = Pipeline([("pre", make_pre(sub[feats])), ("model", est)])
        try:
            sc = cross_val_score(pipe, sub[feats], sub[target], cv=5, scoring=scoring).mean() * 100
        except Exception:
            sc = np.nan
        knn_rows.append({"k": k, "CV %": sc})
    knn = pd.DataFrame(knn_rows)

    clus = None
    if len(feats) >= 2:
        Z = data[feats]
        if len(Z) > 5000:
            Z = Z.sample(5000, random_state=42)
        Xs = make_pre(Z).fit_transform(Z)
        yz = y.loc[Z.index].astype(str).to_numpy()
        if clf:
            k = min(max(int(len(set(yz))), 2), 10)
        else:
            sil = {kk: silhouette_score(Xs, KMeans(kk, n_init=5, random_state=42).fit_predict(Xs),
                                        sample_size=min(len(Xs), 3000), random_state=42)
                   for kk in range(2, 9)}
            k = max(sil, key=sil.get)
        eps = float(np.percentile(NearestNeighbors(n_neighbors=5).fit(Xs).kneighbors(Xs)[0][:, -1], 90))
        algos = {"KMeans": KMeans(k, n_init=10, random_state=42),
                 "Agglomerative": AgglomerativeClustering(k),
                 "Gaussian Mixture": GaussianMixture(k, random_state=42),
                 "DBSCAN": DBSCAN(eps=eps, min_samples=5)}
        crow = []
        for n, a in algos.items():
            lab = a.fit_predict(Xs)
            sc, db, _ = cluster_scores(Xs, lab)
            row = {"Algorithm": n, "Clusters found": len(set(lab) - {-1}),
                   "Silhouette": sc, "Davies-Bouldin": db}
            if clf:
                row["Purity %"] = pd.crosstab(lab, yz).max(axis=1).sum() / len(lab) * 100
                row["Adjusted Rand"] = adjusted_rand_score(yz, lab)
            crow.append(row)
        clus = pd.DataFrame(crow)
    return sup, knn, clus, task


REPORT_CSS = """
body{font-family:'DM Sans',Arial,sans-serif;margin:0;background:#f4f7fb;color:#0B1220}
.wrap{max-width:1050px;margin:0 auto;padding:32px 24px}
header{background:linear-gradient(120deg,#0B1220,#14325a);color:#fff;border-radius:18px;padding:28px 32px}
header h1{margin:0 0 6px;font-size:30px}
header p{margin:0;color:#b8c7e0}
h2{margin:34px 0 10px;font-size:21px;border-left:5px solid #4FA89B;padding-left:10px}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}
.card{flex:1;min-width:150px;background:#fff;border:1px solid #e1e8f2;border-radius:14px;padding:14px 16px}
.card b{display:block;font-size:26px}
.card span{color:#5b6b85;font-size:13px}
table.t{border-collapse:collapse;width:100%;background:#fff;font-size:13px;margin:10px 0}
table.t th{background:#0B1220;color:#fff;text-align:left;padding:8px 10px}
table.t td{padding:7px 10px;border-bottom:1px solid #e8eef6}
.note{color:#5b6b85;font-size:13px}
.ok{color:#0f9d8a}
@media print{body{background:#fff}.wrap{padding:0}}
"""


def build_html_report(name, raw, clean, log, acc, res, cl):
    """Self-contained HTML report. Open it in a browser, or print it to PDF."""
    first = [True]
    esc = htmllib.escape

    def fig_html(fig, h=340):
        style(fig, h)
        fig.update_layout(template="plotly_white", font=dict(family="DM Sans", color="#0B1220"))
        out = fig.to_html(full_html=False, include_plotlyjs="cdn" if first[0] else False)
        first[0] = False
        return out

    def table(d):
        return d.to_html(index=False, border=0, classes="t", float_format=lambda v: f"{v:,.3f}")

    cards = [("Rows", f"{len(raw):,}"), ("Columns", raw.shape[1]),
             ("Missing cells", f"{int(raw.isna().sum().sum()):,}"),
             ("Duplicate rows", f"{int(raw.duplicated().sum()):,}"),
             ("Quality score", f"{quality_score(raw)}/100")]
    h = [f"<!doctype html><html><head><meta charset='utf-8'><title>DataLens report</title>"
         f"<style>{REPORT_CSS}</style></head><body><div class='wrap'>",
         f"<header><h1>DataLens report</h1><p>{esc(str(name))} &middot; generated "
         f"{datetime.now():%d %b %Y, %H:%M}</p></header>",
         "<div class='cards'>" + "".join(f"<div class='card'><b>{v}</b><span>{k}</span></div>"
                                         for k, v in cards) + "</div>"]

    h.append("<h2>Data quality</h2>")
    miss = (raw.isna().mean() * 100).round(2)
    miss = miss[miss > 0].sort_values()
    if miss.empty:
        h.append("<p class='ok'>No missing values found.</p>")
    else:
        h.append(fig_html(px.bar(miss, orientation="h", labels={"value": "% missing", "index": ""}),
                          110 + 30 * len(miss)))
    nd = int(raw.duplicated().sum())
    h.append(f"<p>{nd} duplicate rows found.</p>" if nd else "<p class='ok'>No duplicate rows.</p>")
    oc = outlier_counts(raw)
    oc = oc[oc > 0]
    if len(oc):
        h.append("<p>Outliers by the IQR rule:</p>")
        h.append(table(oc.rename("Outlier rows").rename_axis("Column").reset_index()))
    h.append("<h2>Columns</h2>")
    h.append(table(pd.DataFrame({"Column": raw.columns, "Type": raw.dtypes.astype(str).values,
                                 "Unique": raw.nunique().values,
                                 "Missing %": (raw.isna().mean() * 100).round(2).values})))

    h.append("<h2>Cleaning</h2>")
    if clean is not None and log:
        h.append("<ul>" + "".join(f"<li>{esc(x)}</li>" for x in log) + "</ul>")
        h.append(f"<p>Rows {len(raw):,} to {len(clean):,}. Quality score {quality_score(raw)} "
                 f"to <b>{quality_score(clean)}</b>.</p>")
    else:
        h.append("<p class='note'>Cleaning was not run for this report.</p>")

    if acc:
        sup, knn, clus, atask = acc
        word = "accuracy" if atask == "Classification" else "R² score"
        h.append("<h2>Accuracy of every algorithm</h2>")
        h.append(f"<p>Best algorithm: <b>{esc(sup.iloc[0]['Algorithm'])}</b> with "
                 f"<b>{sup.iloc[0]['Test %']:.1f}%</b> test {word} ({atask.lower()}).</p>")
        fig = px.bar(sup.sort_values("Test %"), x="Test %", y="Algorithm", orientation="h",
                     text_auto=".1f", labels={"Algorithm": ""})
        fig.update_traces(marker_color=TEAL, textposition="outside")
        h.append(fig_html(fig, 100 + 42 * len(sup)))
        h.append(table(sup))
        if knn["CV %"].notna().any():
            kb = knn.loc[knn["CV %"].idxmax()]
            h.append(f"<p>Best KNN setting: k = {int(kb['k'])} ({kb['CV %']:.1f}% cross-validated).</p>")
        if clus is not None:
            h.append("<h2>Clustering scores</h2>")
            h.append(table(clus))
    if res is not None:
        h.append("<h2>Model comparison</h2>")
        h.append(table(res))
    if cl is not None:
        h.append("<h2>Clustering run</h2>")
        h.append(table(cl))
    h.append("<p class='note' style='margin-top:36px'>Created with DataLens.</p></div></body></html>")
    return "".join(h)


def build_excel_report(raw, clean, log, acc, res, cl):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        summary = {"Rows": len(raw), "Columns": raw.shape[1],
                   "Missing cells": int(raw.isna().sum().sum()),
                   "Duplicate rows": int(raw.duplicated().sum()),
                   "Quality score (before)": quality_score(raw)}
        if clean is not None:
            summary["Quality score (after cleaning)"] = quality_score(clean)
            summary["Rows after cleaning"] = len(clean)
        pd.DataFrame({"Metric": list(summary), "Value": list(summary.values())}).to_excel(w, sheet_name="Summary", index=False)
        pd.DataFrame({"Column": raw.columns, "Type": raw.dtypes.astype(str).values,
                      "Unique": raw.nunique().values,
                      "Missing %": (raw.isna().mean() * 100).round(2).values}).to_excel(w, sheet_name="Columns", index=False)
        outlier_counts(raw).rename("Outlier rows").rename_axis("Column").reset_index().to_excel(w, sheet_name="Outliers", index=False)
        if log:
            pd.DataFrame({"Cleaning step": log}).to_excel(w, sheet_name="Cleaning log", index=False)
        if acc:
            sup, knn, clus, _ = acc
            sup.to_excel(w, sheet_name="Accuracy", index=False)
            knn.to_excel(w, sheet_name="KNN by k", index=False)
            if clus is not None:
                clus.to_excel(w, sheet_name="Clustering accuracy", index=False)
        if res is not None:
            res.to_excel(w, sheet_name="Models", index=False)
        if cl is not None:
            cl.to_excel(w, sheet_name="Clustering run", index=False)
        if clean is not None:
            clean.to_excel(w, sheet_name="Cleaned data", index=False)
    return buf.getvalue()


def build_pdf_report(name, raw, clean, log, acc, res, cl):
    """One-click PDF report built with reportlab and matplotlib (no browser needed)."""
    from matplotlib.figure import Figure
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape as xesc

    navy, accent = colors.HexColor("#14325a"), colors.HexColor("#4FA89B")
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=9.5)
    head = ParagraphStyle("hd", parent=small, textColor=colors.white, fontName="Helvetica-Bold")
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=14, textColor=navy, spaceBefore=14, spaceAfter=6)
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=26, textColor=navy, alignment=0, spaceAfter=2)
    W = 18 * cm

    def cell(v, dec):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else f"{v:,.{dec}f}"
        return xesc(str(v))

    def tbl(d, dec=3, max_rows=40):
        d = d.head(max_rows)
        rows = [[Paragraph(xesc(str(c)), head) for c in d.columns]]
        rows += [[Paragraph(cell(v, dec), small) for v in r] for r in d.itertuples(index=False)]
        t = Table(rows, colWidths=[W / len(d.columns)] * len(d.columns), repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), navy),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5fa")]),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d5deea")),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        return t

    def bars(labels, values, ttl, xlabel, color="#4FA89B", fmt="{:.1f}"):
        h = max(2.2, 0.38 * len(labels) + 1.1)
        fig = Figure(figsize=(8, h))
        ax = fig.subplots()
        ax.barh([str(x) for x in labels], values, color=color)
        for i, v in enumerate(values):
            ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_title(ttl, fontsize=11, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        return Image(buf, width=16 * cm, height=16 * cm * h / 8)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6b7a90"))
        canvas.drawString(1.5 * cm, 1 * cm, "DataLens report")
        canvas.drawRightString(A4[0] - 1.5 * cm, 1 * cm, f"Page {doc.page}")
        canvas.restoreState()

    S = []
    S.append(Paragraph("DataLens report", title))
    S.append(Paragraph(f"{xesc(str(name))} | generated {datetime.now():%d %b %Y, %H:%M}", body))
    S.append(Spacer(1, 10))
    stats = [("Rows", f"{len(raw):,}"), ("Columns", str(raw.shape[1])),
             ("Missing cells", f"{int(raw.isna().sum().sum()):,}"),
             ("Duplicate rows", f"{int(raw.duplicated().sum()):,}"),
             ("Quality score", f"{quality_score(raw)}/100")]
    big = ParagraphStyle("big", parent=body, fontSize=16, leading=20, fontName="Helvetica-Bold", textColor=navy)
    cards = Table([[Paragraph(v, big) for _, v in stats], [Paragraph(k, small) for k, _ in stats]],
                  colWidths=[W / 5] * 5)
    cards.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, accent),
                               ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d5deea")),
                               ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5fa")),
                               ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    S.append(cards)

    S.append(Paragraph("Data quality", h2))
    miss = (raw.isna().mean() * 100).round(2)
    miss = miss[miss > 0].sort_values()
    if miss.empty:
        S.append(Paragraph("No missing values found.", body))
    else:
        S.append(bars(list(miss.index), list(miss.values), "Missing values by column", "% missing",
                      "#C9A24D", "{:.1f}%"))
    nd = int(raw.duplicated().sum())
    S.append(Paragraph(f"{nd} duplicate rows found." if nd else "No duplicate rows.", body))
    oc = outlier_counts(raw)
    oc = oc[oc > 0]
    if len(oc):
        S.append(Spacer(1, 6))
        S.append(Paragraph("Outliers by the IQR rule:", body))
        S.append(tbl(oc.rename("Outlier rows").rename_axis("Column").reset_index(), 0))
    S.append(Paragraph("Columns", h2))
    S.append(tbl(pd.DataFrame({"Column": raw.columns, "Type": raw.dtypes.astype(str).values,
                               "Unique": raw.nunique().values,
                               "Missing %": (raw.isna().mean() * 100).round(2).values}), 2, 60))

    S.append(Paragraph("Cleaning", h2))
    if clean is not None and log:
        for x in log:
            S.append(Paragraph("- " + xesc(x), body))
        S.append(Paragraph(f"Rows {len(raw):,} to {len(clean):,}. Quality score {quality_score(raw)} "
                           f"to <b>{quality_score(clean)}</b>.", body))
    else:
        S.append(Paragraph("Cleaning was not run for this report.", body))

    if acc:
        sup, knn, clus, atask = acc
        word = "accuracy" if atask == "Classification" else "R2 score"
        S.append(Paragraph("Accuracy of every algorithm", h2))
        S.append(Paragraph(f"Best algorithm: <b>{xesc(str(sup.iloc[0]['Algorithm']))}</b> with "
                           f"<b>{sup.iloc[0]['Test %']:.1f}%</b> test {word} ({atask.lower()}).", body))
        s2 = sup.iloc[::-1]
        S.append(bars(list(s2["Algorithm"]), list(s2["Test %"]), f"Test {word} by algorithm (%)",
                      "%"))
        S.append(tbl(sup, 1))
        if knn["CV %"].notna().any():
            kb = knn.loc[knn["CV %"].idxmax()]
            S.append(Spacer(1, 6))
            S.append(Paragraph(f"Best KNN setting: k = {int(kb['k'])} ({kb['CV %']:.1f}% cross-validated).", body))
        if clus is not None:
            S.append(Paragraph("Clustering scores", h2))
            S.append(tbl(clus, 3))
    if res is not None:
        S.append(Paragraph("Model comparison", h2))
        S.append(tbl(res, 3))
    if cl is not None:
        S.append(Paragraph("Clustering run", h2))
        S.append(tbl(cl, 3))

    out = io.BytesIO()
    SimpleDocTemplate(out, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm,
                      bottomMargin=1.8 * cm, title="DataLens report").build(
        S, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue()


# ----------------------------------------------------------------------------
# Hero + data intake
# ----------------------------------------------------------------------------
st.markdown(
    """<div class="hero"><span class="pill">Data Science</span><span class="pill">Machine Learning</span>
<h1>DataLens</h1>
<p>Upload a dataset. See what is wrong with it, fix it, and find out which model learns it best,
all in one place.</p></div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Dataset")
    up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    sample = st.selectbox("Or try a sample", ["None", "Breast cancer (messy: nulls + duplicates)",
                                              "Wine (classification)", "Diabetes (regression)"])
    st.caption("Built with Python, pandas, scikit-learn, Plotly and Streamlit.")

raw = None
if up is not None:
    try:
        raw = read_file(up)
    except Exception as e:
        st.error(f"Could not read that file ({e}). Check that it is a valid CSV or Excel file.")
elif sample != "None":
    raw = load_sample(sample)

if raw is None:
    st.info("Upload a CSV or Excel file in the sidebar, or pick a sample dataset to begin.")
    st.stop()

raw.columns = [str(c) for c in raw.columns]
src_id = (up.name if up is not None else sample, raw.shape)
if st.session_state.get("src_id") != src_id:
    st.session_state.update(src_id=src_id, clean=None, log=None, results=None, cl_results=None, acc=None, report=None)

df = st.session_state.get("clean")
working = df if df is not None else raw

tab_over, tab_q, tab_fix, tab_eda, tab_acc, tab_ml, tab_cl, tab_rep = st.tabs(
    ["Overview", "Data quality", "Clean", "Explore", "Accuracy", "Models", "Clustering", "Report"])

# ----------------------------- Overview -------------------------------------
with tab_over:
    c1, c2 = st.columns([1, 3])
    score = quality_score(raw)
    color = TEAL if score >= 85 else AMBER if score >= 65 else CORAL
    with c1:
        st.markdown(f'<div class="score" style="color:{color}">{score}</div>'
                    '<div class="scorelabel">Data quality score out of 100</div>',
                    unsafe_allow_html=True)
        if df is not None:
            st.markdown(f"After cleaning: **{quality_score(df)}**")
    with c2:
        m = st.columns(4)
        m[0].metric("Rows", f"{len(raw):,}")
        m[1].metric("Columns", raw.shape[1])
        m[2].metric("Missing cells", f"{int(raw.isna().sum().sum()):,}")
        m[3].metric("Duplicate rows", f"{int(raw.duplicated().sum()):,}")
    st.subheader("Preview")
    st.dataframe(raw.head(100), width="stretch")
    st.subheader("Column types")
    info = pd.DataFrame({"type": raw.dtypes.astype(str), "unique": raw.nunique(),
                         "missing %": (raw.isna().mean() * 100).round(2)})
    st.dataframe(info, width="stretch")
    st.subheader("Summary statistics")
    st.dataframe(raw.describe(include="all").T, width="stretch")

# ----------------------------- Data quality ---------------------------------
with tab_q:
    miss = (raw.isna().mean() * 100).round(2)
    miss = miss[miss > 0].sort_values()
    a, b = st.columns(2)
    with a:
        st.subheader("Missing values")
        if miss.empty:
            st.success("No missing values found.")
        else:
            fig = px.bar(miss, orientation="h", labels={"value": "% missing", "index": ""})
            st.plotly_chart(style(fig, 100 + 32 * len(miss)), width="stretch")
    with b:
        st.subheader("Outliers (IQR rule)")
        oc = outlier_counts(raw)
        oc = oc[oc > 0].sort_values()
        if oc.empty:
            st.success("No outliers flagged.")
        else:
            fig = px.bar(oc, orientation="h", labels={"value": "outlier rows", "index": ""})
            fig.update_traces(marker_color=AMBER)
            st.plotly_chart(style(fig, 100 + 32 * len(oc)), width="stretch")
    st.subheader("Duplicate rows")
    dups = raw[raw.duplicated(keep=False)].sort_values(list(raw.columns[:2]))
    if dups.empty:
        st.success("No duplicate rows.")
    else:
        st.warning(f"{int(raw.duplicated().sum())} rows repeat an earlier row.")
        st.dataframe(dups.head(200), width="stretch")

# ----------------------------- Clean ----------------------------------------
with tab_fix:
    st.subheader("Fix the dataset")
    c = st.columns(4)
    drop_dups = c[0].checkbox("Remove duplicates", True)
    thresh = c[1].slider("Drop columns above % missing", 10, 100, 60)
    num_s = c[2].selectbox("Numeric gaps", ["Median", "Mean", "KNN (k=5)", "Drop rows"])
    cat_s = c[3].selectbox("Text gaps", ["Mode", "Fill with 'Missing'", "Drop rows"])
    if st.button("Clean dataset", type="primary"):
        st.session_state.clean, st.session_state.log = clean_data(raw, drop_dups, thresh, num_s, cat_s)
        st.session_state.results = st.session_state.cl_results = st.session_state.acc = None
        st.rerun()
    if st.session_state.get("log"):
        for line in st.session_state.log:
            st.success(line)
        d = st.session_state.clean
        k = st.columns(3)
        k[0].metric("Rows", f"{len(d):,}", f"{len(d) - len(raw):,}")
        k[1].metric("Missing cells", int(d.isna().sum().sum()), -int(raw.isna().sum().sum()))
        k[2].metric("Quality score", quality_score(d), round(quality_score(d) - quality_score(raw), 1))
        st.dataframe(d.head(100), width="stretch")
        st.download_button("Download cleaned CSV", d.to_csv(index=False).encode(),
                           "cleaned_data.csv", "text/csv")
    else:
        st.caption("Models and clustering use the cleaned data once you run this step. "
                   "Otherwise they impute on their own.")

# ----------------------------- Explore --------------------------------------
with tab_eda:
    num_cols = working.select_dtypes("number").columns.tolist()
    st.subheader("Distributions")
    if num_cols:
        col = st.selectbox("Column", working.columns)
        fig = (px.histogram(working, x=col, marginal="box", nbins=40)
               if col in num_cols else px.bar(working[col].value_counts().head(25)))
        st.plotly_chart(style(fig, 420), width="stretch")
    if len(num_cols) >= 2:
        st.subheader("Correlation")
        corr = working[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        aspect="auto")
        st.plotly_chart(style(fig, 150 + 38 * len(num_cols)), width="stretch")
        st.subheader("Relationship")
        x, y = st.columns(2)
        xv = x.selectbox("X", num_cols, 0)
        yv = y.selectbox("Y", num_cols, min(1, len(num_cols) - 1))
        st.plotly_chart(style(px.scatter(working, x=xv, y=yv, opacity=.7), 420), width="stretch")

# ----------------------------- Models ---------------------------------------
with tab_ml:
    st.subheader("Supervised models")
    cols = working.columns.tolist()
    t1, t2, t3 = st.columns([2, 2, 1])
    target = t1.selectbox("Target column", cols, index=len(cols) - 1)
    ttype = t2.radio("Task", ["Auto", "Classification", "Regression"], horizontal=True)
    test_size = t3.slider("Test split", 0.1, 0.5, 0.2, 0.05)
    auto = ("Classification" if (working[target].dtype == object or working[target].nunique() <= 15)
            else "Regression")
    task = auto if ttype == "Auto" else ttype
    if task == "Regression" and not pd.api.types.is_numeric_dtype(working[target]):
        st.error("Regression needs a numeric target. Switch to Classification or pick another column.")
    feats = st.multiselect("Features", [c for c in cols if c != target],
                           default=[c for c in cols if c != target])
    zoo = CLF if task == "Classification" else REG
    chosen = st.multiselect("Algorithms", list(zoo), default=list(zoo))
    st.caption(f"Detected task: **{task}**")
    if st.button("Train and compare", type="primary", disabled=not (feats and chosen)
                 or (task == "Regression" and not pd.api.types.is_numeric_dtype(working[target]))):
        res, fit = run_models(working, target, feats, task, test_size, chosen)
        st.session_state.results = (res, fit, task)
    if st.session_state.get("results"):
        res, fit, rtask = st.session_state.results
        key = "Accuracy" if rtask == "Classification" else "R2"
        best = res.iloc[0]
        k = st.columns(3)
        k[0].metric("Best model", best["Model"])
        k[1].metric(key, f"{best[key]:.3f}")
        k[2].metric("Models compared", len(res))
        st.dataframe(res.style.format({c: "{:.3f}" for c in res.columns if c != "Model"})
                     .background_gradient(subset=[key], cmap="GnBu"), width="stretch", hide_index=True)
        score_cols = [c for c in res.columns if c not in ("Model", "Train time (s)", "MAE", "RMSE")]
        fig = px.bar(res.melt("Model", score_cols), x="Model", y="value", color="variable",
                     barmode="group", labels={"value": "score", "variable": ""})
        st.plotly_chart(style(fig, 420), width="stretch")
        st.subheader("Inspect one model")
        pick = st.selectbox("Model", res["Model"])
        pipe, yte, pred = fit[pick]
        a, b = st.columns(2)
        with a:
            if rtask == "Classification":
                labels = sorted(pd.unique(yte))
                cm = confusion_matrix(yte, pred, labels=labels)
                fig = px.imshow(cm, x=[str(x) for x in labels], y=[str(x) for x in labels],
                                text_auto=True, color_continuous_scale="Teal",
                                labels=dict(x="Predicted", y="Actual"))
                fig.update_layout(title="Confusion matrix")
            else:
                fig = px.scatter(x=yte, y=pred, labels=dict(x="Actual", y="Predicted"), opacity=.7)
                lo, hi = float(min(yte.min(), pred.min())), float(max(yte.max(), pred.max()))
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                         line=dict(color=AMBER, dash="dash"), name="perfect"))
                fig.update_layout(title="Actual vs predicted")
            st.plotly_chart(style(fig, 400), width="stretch")
        with b:
            imp = importances(pipe)
            if imp is not None:
                fig = px.bar(imp, orientation="h", labels={"value": "importance", "index": ""})
                fig.update_layout(title="Top features")
                st.plotly_chart(style(fig, 400), width="stretch")
            else:
                st.caption("This model does not expose feature importances.")

# ----------------------------- Clustering -----------------------------------
with tab_cl:
    st.subheader("Unsupervised clustering")
    ccols = [c for c in working.columns
             if pd.api.types.is_numeric_dtype(working[c]) or working[c].nunique() <= 30]
    use = st.multiselect("Features", ccols, default=ccols[:12])
    st.caption("Text columns are one-hot encoded and everything is scaled automatically. "
               "Columns with more than 30 unique text values (like timestamps) are left out.")
    k = st.slider("Number of clusters (KMeans, Agglomerative, GMM)", 2, 10, 3)
    e1, e2 = st.columns(2)
    eps = e1.slider("DBSCAN eps", 0.1, 5.0, 1.5, 0.1)
    ms = e2.slider("DBSCAN min samples", 2, 20, 5)
    if st.button("Run clustering", type="primary", disabled=len(use) < 2):
        Z = working[use]
        if len(Z) > 5000:
            Z = Z.sample(5000, random_state=42)
        Xs = make_pre(Z).fit_transform(Z)
        algos = {"KMeans": KMeans(k, n_init=10, random_state=42),
                 "Agglomerative": AgglomerativeClustering(k),
                 "Gaussian Mixture": GaussianMixture(k, random_state=42),
                 "DBSCAN": DBSCAN(eps=eps, min_samples=ms)}
        rows, labs = [], {}
        for n, a in algos.items():
            lab = a.fit_predict(Xs)
            s, db, ch = cluster_scores(Xs, lab)
            labs[n] = lab
            rows.append({"Algorithm": n, "Clusters found": len(set(lab) - {-1}),
                         "Silhouette (higher)": s, "Davies-Bouldin (lower)": db,
                         "Calinski-Harabasz (higher)": ch})
        sweep = []
        for kk in range(2, 11):
            km = KMeans(kk, n_init=10, random_state=42).fit(Xs)
            sweep.append({"k": kk, "Inertia": km.inertia_,
                          "Silhouette": silhouette_score(Xs, km.labels_,
                                                         sample_size=min(len(Xs), 3000), random_state=42)})
        st.session_state.cl_results = (pd.DataFrame(rows), labs, PCA(2).fit_transform(Xs),
                                       pd.DataFrame(sweep))
    if st.session_state.get("cl_results"):
        tbl, labs, pts, sweep = st.session_state.cl_results
        st.dataframe(tbl.style.format({c: "{:.3f}" for c in tbl.columns[2:]}), width="stretch",
                     hide_index=True)
        a, b = st.columns(2)
        with a:
            fig = px.line(sweep, x="k", y="Inertia", markers=True, title="Elbow curve")
            st.plotly_chart(style(fig, 330), width="stretch")
        with b:
            fig = px.line(sweep, x="k", y="Silhouette", markers=True, title="Silhouette by k")
            st.plotly_chart(style(fig, 330), width="stretch")
        pick = st.selectbox("View clusters from", list(labs))
        fig = px.scatter(x=pts[:, 0], y=pts[:, 1], color=labs[pick].astype(str),
                         labels=dict(x="PC1", y="PC2", color="cluster"), opacity=.8)
        st.plotly_chart(style(fig, 460), width="stretch")
        st.caption("DBSCAN labels noise points as -1.")

# ----------------------------- Accuracy -------------------------------------
with tab_acc:
    st.subheader("Accuracy of every algorithm")
    st.caption("Classification is scored by accuracy. Regression is scored by R², its equivalent. "
               "Clustering is compared with the true labels when the target is a category.")
    acols = working.columns.tolist()
    q1, q2, q3 = st.columns([2, 2, 1])
    a_target = q1.selectbox("Target column", acols, index=len(acols) - 1, key="acc_target")
    a_type = q2.radio("Task", ["Auto", "Classification", "Regression"], horizontal=True, key="acc_task")
    a_split = q3.slider("Test split", 0.1, 0.5, 0.2, 0.05, key="acc_split")
    a_auto = ("Classification" if (working[a_target].dtype == object or working[a_target].nunique() <= 15)
              else "Regression")
    a_task = a_auto if a_type == "Auto" else a_type
    a_bad = a_task == "Regression" and not pd.api.types.is_numeric_dtype(working[a_target])
    if a_bad:
        st.error("Regression needs a numeric target. Switch to Classification or pick another column.")
    st.caption(f"Detected task: **{a_task}**")
    if st.button("Calculate accuracy for all algorithms", type="primary", disabled=a_bad, key="acc_run"):
        with st.spinner("Training every algorithm. This can take a minute on larger datasets."):
            st.session_state.acc = accuracy_suite(working, a_target, a_task, a_split)

    if st.session_state.get("acc"):
        sup, knn, clus, atask = st.session_state.acc
        word = "accuracy" if atask == "Classification" else "R² score"
        top = sup.iloc[0]
        kb = knn.loc[knn["CV %"].idxmax()] if knn["CV %"].notna().any() else None
        m = st.columns(4)
        m[0].metric("Best algorithm", top["Algorithm"])
        m[1].metric(f"Test {word}", f"{top['Test %']:.1f}%")
        m[2].metric("Algorithms tested", len(sup) + (len(clus) if clus is not None else 0))
        m[3].metric("Best KNN k", int(kb["k"]) if kb is not None else "n/a")

        st.markdown(f"#### Supervised algorithms ({atask.lower()})")
        fig = px.bar(sup.sort_values("Test %"), x="Test %", y="Algorithm", orientation="h",
                     text_auto=".1f", labels={"Test %": f"Test {word} (%)", "Algorithm": ""})
        fig.update_traces(marker_color=TEAL, textposition="outside")
        st.plotly_chart(style(fig, 90 + 46 * len(sup)), width="stretch")
        st.dataframe(sup.style.format({c: "{:.1f}%" for c in ["Train %", "Test %", "CV %"]}
                                      | {"Overfit gap": "{:.1f}"})
                     .background_gradient(subset=["Test %"], cmap="GnBu"),
                     width="stretch", hide_index=True)
        st.caption("Overfit gap is train minus test. A large gap means the model memorised the training data.")
        fig = px.bar(sup.melt("Algorithm", ["Train %", "Test %", "CV %"]), x="Algorithm", y="value",
                     color="variable", barmode="group", labels={"value": "score (%)", "variable": ""},
                     title="Train vs test vs cross-validation")
        st.plotly_chart(style(fig, 400), width="stretch")

        st.markdown("#### KNN: score for each k")
        fig = px.line(knn, x="k", y="CV %", markers=True, labels={"CV %": f"CV {word} (%)"})
        if kb is not None:
            fig.add_trace(go.Scatter(x=[kb["k"]], y=[kb["CV %"]], mode="markers", name="best k",
                                     marker=dict(size=14, color=AMBER)))
        st.plotly_chart(style(fig, 340), width="stretch")

        if clus is not None:
            st.markdown("#### Clustering algorithms")
            if "Purity %" in clus:
                st.caption("Purity is the share of points sitting in a cluster dominated by their true class. "
                           "It works as clustering accuracy.")
                fmt = {"Silhouette": "{:.3f}", "Davies-Bouldin": "{:.3f}",
                       "Purity %": "{:.1f}%", "Adjusted Rand": "{:.3f}"}
                fig = px.bar(clus, x="Algorithm", y="Purity %", text_auto=".1f")
            else:
                st.caption("No category labels here, so clustering is scored by silhouette (higher is better).")
                fmt = {"Silhouette": "{:.3f}", "Davies-Bouldin": "{:.3f}"}
                fig = px.bar(clus, x="Algorithm", y="Silhouette", text_auto=".3f")
            fig.update_traces(marker_color=VIOLET)
            st.plotly_chart(style(fig, 340), width="stretch")
            st.dataframe(clus.style.format(fmt), width="stretch", hide_index=True)

            if "Purity %" in clus:
                st.markdown("#### All algorithms ranked")
                allr = pd.concat([
                    sup[["Algorithm", "Test %"]].rename(columns={"Test %": "Score %"}).assign(Family=atask),
                    clus[["Algorithm", "Purity %"]].rename(columns={"Purity %": "Score %"})
                    .assign(Algorithm=lambda d: d["Algorithm"] + " (clustering)", Family="Clustering")])
                fig = px.bar(allr.sort_values("Score %"), x="Score %", y="Algorithm", color="Family",
                             orientation="h", labels={"Algorithm": ""})
                st.plotly_chart(style(fig, 120 + 34 * len(allr)), width="stretch")

# ----------------------------- Report ---------------------------------------
with tab_rep:
    st.subheader("Download report")
    st.caption("The report covers whatever you have run so far. Run Clean, Accuracy, Models or "
               "Clustering first to include them, then build the report.")
    acc_r = st.session_state.get("acc")
    mod_r = st.session_state.get("results")
    clu_r = st.session_state.get("cl_results")
    checks = [("Data quality audit", True), ("Cleaning steps", df is not None),
              ("Accuracy of all algorithms", bool(acc_r)), ("Model comparison", bool(mod_r)),
              ("Clustering results", bool(clu_r))]
    for label, ok in checks:
        st.markdown(f"{'✅' if ok else '⬜'} {label}" + ("" if ok else " (not run yet)"))
    if st.button("Build report", type="primary"):
        with st.spinner("Building report..."):
            nm, lg = src_id[0], st.session_state.get("log")
            args = (raw, df, lg, acc_r, mod_r[0] if mod_r else None, clu_r[0] if clu_r else None)
            try:
                pdf = build_pdf_report(nm, *args)
            except Exception as e:
                pdf = None
                st.warning(f"The PDF could not be built ({e}). The HTML and Excel reports still work.")
            st.session_state.report = (build_html_report(nm, *args), build_excel_report(*args), pdf)
    if st.session_state.get("report"):
        html_doc, xlsx, pdf = st.session_state.report
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download PDF report", pdf or b"", "datalens_report.pdf", "application/pdf",
                           width="stretch", type="primary", disabled=pdf is None)
        d2.download_button("Download HTML report", html_doc.encode("utf-8"), "datalens_report.html",
                           "text/html", width="stretch")
        d3.download_button("Download Excel report", xlsx, "datalens_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           width="stretch")