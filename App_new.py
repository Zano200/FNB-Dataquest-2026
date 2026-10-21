# Credit Risk Dashboard
# Run: streamlit run App.py
# Requires: loan_book.csv, improved_pipe.pkl, woe_artefacts.pkl, train_stats.pkl

import warnings; warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Risk Dashboard", page_icon="💳", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0E0F0D; color: #E8EBE4; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2E312D 20%, #8B5800 40%, #10ADAE 100%);
}
section[data-testid="stSidebar"] * { color: white !important; }
div[data-testid="metric-container"] {
    background-color: #161714;
    border: 1px solid #2E312D;
    border-left: 3px solid #1D9E75;
    border-radius: 8px;
    padding: 12px;
}
div[data-testid="metric-container"] label { color: #9BA396 !important; font-size: 11px; }
div[data-testid="metric-container"] div[data-testid="metric-value"] { color: #E8EBE4 !important; }
h1, h2, h3 { color: #E8EBE4 !important; }
.stDataFrame { background-color: #161714; }
.stSelectbox label, .stSlider label, .stRadio label { color: #9BA396 !important; }
.stButton > button {
    background-color: #1D9E75; color: white; border: none;
    border-radius: 6px; font-weight: 500;
}
.stButton > button:hover { background-color: #0D6E52; }
</style>
""", unsafe_allow_html=True)

TEAL = '#1D9E75'; CORAL = '#D85A30'; AMBER = '#EF9F27'; BLUE = '#378ADD'; GRAY = '#6B6B6B'
PLOT_LAYOUT = dict(
    paper_bgcolor='#161714', plot_bgcolor='#161714',
    font=dict(color='#9BA396', size=11),
    margin=dict(l=10, r=10, t=36, b=10),
    xaxis=dict(gridcolor='#2E312D', zerolinecolor='#2E312D'),
    yaxis=dict(gridcolor='#2E312D', zerolinecolor='#2E312D'),
)

# ── LOAD ARTEFACTS ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    woe   = joblib.load('woe_artefacts.pkl')
    stats = joblib.load('train_stats.pkl')
    return woe, stats

@st.cache_data
def load_data():
    df = pd.read_csv('loan_book.csv')
    return df

loan_df = load_data()


# ── PIPELINE HELPERS (must match notebook exactly) ───────────────────────────
def clean_dataframe(df, ref_df=None):
    d = df.copy()
    r = ref_df if ref_df is not None else df
    d['home_ownership'] = d['home_ownership'].str.strip().str.upper().replace({'RENTING':'RENT','OWNER':'OWN'})
    d['loan_purpose']   = d['loan_purpose'].str.strip().str.lower().str.replace(' ','_')
    d['application_date'] = pd.to_datetime(d['application_date'], format='mixed', dayfirst=False, errors='coerce')
    d['ever_delinquent'] = d['months_since_last_delinquency'].notna().astype(int)
    for col in ['annual_income','employment_length_years','num_open_accounts']:
        mv = r[col].median()
        d[f'{col}_missing'] = d[col].isna().astype(int)
        d[col] = d[col].fillna(mv)
    for col in ['annual_income','loan_amount','total_revolving_balance','credit_utilisation_pct']:
        cap = r[col].quantile(0.99)
        d[col] = d[col].clip(upper=cap)
    drop_cols = ['applicant_id_hash','application_date','branch_code_id','set','months_since_last_delinquency']
    return d.drop(columns=[c for c in drop_cols if c in d.columns])

def apply_woe(series, feature, woe_artefacts, categorical=False):
    if categorical:
        return series.map(woe_artefacts['woe_cat_maps'][feature]).fillna(0.0)
    else:
        edges = woe_artefacts['woe_bin_edges'][feature]   # plain float list
        c2w   = dict(enumerate(woe_artefacts['woe_mappings'][feature]))
        codes = pd.cut(series, bins=edges, labels=False, include_lowest=True)
        return codes.map(c2w).fillna(0.0)

def build_features(df, woe_artefacts):
    d = df.copy(); EPS = 1e-6
    d['log_annual_income']           = np.log1p(d['annual_income'])
    d['log_loan_amount']             = np.log1p(d['loan_amount'])
    d['log_total_revolving_balance'] = np.log1p(d['total_revolving_balance'])
    d['loan_to_income']  = d['loan_amount'] / (d['annual_income'] + EPS)
    d['loan_times_dti']  = d['loan_amount'] * d['dti_ratio']
    for col in woe_artefacts['woe_num']:
        d[f'{col}_woe'] = apply_woe(d[col], col, woe_artefacts, categorical=False)
    for col in woe_artefacts['woe_cat']:
        d[f'{col}_woe'] = apply_woe(d[col], col, woe_artefacts, categorical=True)
    d['dti_x_credit_util_woe'] = d['dti_ratio_woe'] * d['credit_utilisation_pct_woe']
    return d

# ── STARTUP: load artefacts & train model (all helpers defined above) ─────────
try:
    woe_artefacts, train_stats = load_artefacts()
    ARTEFACTS_OK = True
except FileNotFoundError:
    ARTEFACTS_OK = False


@st.cache_resource
def train_model(_woe_artefacts):
    """Train locally — avoids sklearn version pickle issues."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    train_raw = loan_df[loan_df['set'] == 'train'].copy().reset_index(drop=True)
    train_c   = clean_dataframe(train_raw)
    train_fe  = build_features(train_c, _woe_artefacts)
    feats     = _woe_artefacts['model_features']
    X_train   = train_fe[feats]
    y_train   = train_fe['default_flag']
    pipe = Pipeline([
        ('sc', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000, C=0.1,
                                  class_weight='balanced', random_state=42))
    ])
    pipe.fit(X_train, y_train)
    return pipe


if ARTEFACTS_OK:
    improved_pipe = train_model(woe_artefacts)


@st.cache_data
def get_test_predictions():
    train_raw = loan_df[loan_df['set']=='train'].copy().reset_index(drop=True)
    test_raw  = loan_df[loan_df['set']=='test'].copy().reset_index(drop=True)
    train_c   = clean_dataframe(train_raw)
    test_c    = clean_dataframe(test_raw, ref_df=train_raw)
    test_fe   = build_features(test_c, woe_artefacts)
    feats     = woe_artefacts['model_features']
    X_test    = test_fe[feats]
    y_test    = test_fe['default_flag']
    proba     = improved_pipe.predict_proba(X_test)[:, 1]
    return y_test.values, proba

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.image("FNB_logo.png", use_container_width=True)
st.sidebar.title("💳 Credit Risk")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "📊 Dataset Overview",
    "📈 EDA",
    "🔢 WoE / IV Analysis",
    "📉 Model Performance",
    "💼 Business Dashboard",
    "🧾 Credit Scorer",
])
st.sidebar.markdown("---")
st.sidebar.caption("Logistic Regression · AUC 0.795 · 120,960 applications \n\nData Scientists: A.S. Gwebityala, S.B. Mda, & B. Teta \n\n2026")

# ── DATASET OVERVIEW ─────────────────────────────────────────────────────────
if page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")

    train_df = loan_df[loan_df['set']=='train']
    test_df  = loan_df[loan_df['set']=='test']

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total rows",        f"{len(loan_df):,}")
    c2.metric("Train rows",        f"{len(train_df):,}")
    c3.metric("Test rows",         f"{len(test_df):,}")
    c4.metric("Default rate (train)", f"{train_df['default_flag'].mean():.1%}")
    c5.metric("Missing values",    f"{loan_df.isnull().sum().sum():,}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data preview")
        st.dataframe(loan_df.head(8), use_container_width=True)

    with col2:
        st.subheader("Missing values by column")
        miss = loan_df.isnull().mean().mul(100).round(2)
        miss = miss[miss > 0].sort_values(ascending=False).reset_index()
        miss.columns = ['Column', 'Missing %']
        if miss.empty:
            st.info("No missing values in raw data (some are encoded as NaN).")
            # Show the MNAR column manually
            miss = pd.DataFrame({'Column':['months_since_last_delinquency'], 'Missing %':[loan_df['months_since_last_delinquency'].isnull().mean()*100]})
        fig = px.bar(miss, x='Missing %', y='Column', orientation='h', color='Missing %',
                     color_continuous_scale=[[0,TEAL],[1,CORAL]])
        fig.update_layout(**PLOT_LAYOUT, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Target distribution — default flag")
    col1, col2 = st.columns(2)
    with col1:
        counts = loan_df.groupby(['set','default_flag']).size().reset_index(name='n')
        counts['label'] = counts['default_flag'].map({0:'No default',1:'Default'})
        fig = px.bar(counts, x='set', y='n', color='label',
                     color_discrete_map={'No default':TEAL,'Default':CORAL},
                     barmode='group', title="Default counts by split")
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        dr = loan_df.groupby('set')['default_flag'].mean().mul(100).reset_index()
        dr.columns = ['Set','Default rate %']
        fig = px.bar(dr, x='Set', y='Default rate %',
                     color_discrete_sequence=[AMBER], title="Default rate by split")
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary statistics")
    num_cols = loan_df.select_dtypes(include='number').columns.tolist()
    st.dataframe(loan_df[num_cols].describe().T.round(3), use_container_width=True)

# ── EDA ──────────────────────────────────────────────────────────────────────
elif page == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")

    num_cols = [c for c in loan_df.select_dtypes(include='number').columns
                if c not in ['default_flag']]

    col1, col2 = st.columns([1,3])
    with col1:
        feat = st.selectbox("Select feature", num_cols, index=num_cols.index('interest_rate') if 'interest_rate' in num_cols else 0)
    
    d0 = loan_df[loan_df['default_flag']==0][feat].dropna()
    d1 = loan_df[loan_df['default_flag']==1][feat].dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean (no default)", f"{d0.mean():.2f}")
    c2.metric("Mean (default)",    f"{d1.mean():.2f}", delta=f"{d1.mean()-d0.mean():+.2f}")
    c3.metric("Median (no default)",f"{d0.median():.2f}")
    c4.metric("Skewness",          f"{loan_df[feat].skew():.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution by default status")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=d0, name='No default', nbinsx=40,
                                   marker_color=TEAL, opacity=0.65, histnorm='probability density'))
        fig.add_trace(go.Histogram(x=d1, name='Default', nbinsx=40,
                                   marker_color=CORAL, opacity=0.65, histnorm='probability density'))
        fig.update_layout(**PLOT_LAYOUT, barmode='overlay',
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Default rate by decile")
        tmp = loan_df[[feat,'default_flag']].dropna()
        tmp['decile'] = pd.qcut(tmp[feat], q=8, duplicates='drop')
        biv = tmp.groupby('decile', observed=True)['default_flag'].agg(['mean','count']).reset_index()
        biv['bin_str'] = biv['decile'].astype(str).str.replace(r'[\(\)\[\]]','',regex=True)
        colours = [CORAL if v>0.2 else AMBER if v>0.12 else TEAL for v in biv['mean']]
        fig = go.Figure(go.Bar(x=biv['bin_str'], y=(biv['mean']*100).round(1),
                               marker_color=colours, text=(biv['mean']*100).round(1),
                               texttemplate='%{text}%', textposition='outside'))
        fig.update_layout(**PLOT_LAYOUT, yaxis_title='Default rate %')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation heatmap")
    corr_cols = ['age','annual_income','employment_length_years','dti_ratio',
                 'credit_utilisation_pct','loan_amount','interest_rate',
                 'num_delinquencies_2yr','num_hard_inquiries_6mo',
                 'months_since_oldest_account','pct_accounts_current','default_flag']
    corr = loan_df[corr_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect='auto',
                    color_continuous_scale='RdBu_r', zmin=-0.6, zmax=0.6)
    fig.update_layout(**PLOT_LAYOUT, height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Categorical features — default rate")
    cat_cols = ['home_ownership','loan_purpose','region','email_domain_type','application_dow']
    available_cat = [c for c in cat_cols if c in loan_df.columns]
    cat_feat = st.selectbox("Select categorical feature", available_cat)
    rates = (loan_df.groupby(cat_feat)['default_flag']
             .agg(['mean','count']).rename(columns={'mean':'default_rate','count':'n'})
             .sort_values('default_rate').reset_index())
    fig = px.bar(rates, x='default_rate', y=cat_feat, orientation='h',
                 color='default_rate', color_continuous_scale=[[0,TEAL],[0.5,AMBER],[1,CORAL]],
                 text=rates['default_rate'].mul(100).round(1).astype(str)+'%')
    fig.update_layout(**PLOT_LAYOUT, showlegend=False, coloraxis_showscale=False,
                      xaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)

    # ── Bivariate scatterplot ─────────────────────────────────────────────────
    st.subheader("Bivariate explorer — scatterplot")
    st.caption("Coloured by default status. Sample of 3,000 points for performance.")

    scatter_num = ['annual_income','dti_ratio','credit_utilisation_pct',
                   'interest_rate','age','employment_length_years',
                   'loan_amount','months_since_oldest_account','num_delinquencies_2yr']
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        x_feat = st.selectbox("X axis", scatter_num, index=0)
    with sc2:
        y_feat = st.selectbox("Y axis", scatter_num, index=1)
    with sc3:
        jitter = st.checkbox("Add jitter (discrete features)", value=False)

    samp = loan_df[['default_flag', x_feat, y_feat]].dropna().sample(
        n=min(3000, len(loan_df)), random_state=42
    ).copy()
    samp['Default'] = samp['default_flag'].map({0: 'No default', 1: 'Default'})

    if jitter:
        samp[x_feat] = samp[x_feat] + np.random.uniform(-0.3, 0.3, len(samp))
        samp[y_feat] = samp[y_feat] + np.random.uniform(-0.3, 0.3, len(samp))

    fig = px.scatter(
        samp, x=x_feat, y=y_feat, color='Default',
        color_discrete_map={'No default': TEAL, 'Default': CORAL},
        opacity=0.45, marginal_x='histogram', marginal_y='histogram',
    )
    fig.update_traces(marker=dict(size=4), selector=dict(type='scatter'))
    fig.update_layout(**PLOT_LAYOUT, height=520,
                      legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    # Insight callout
    corr_val = samp[[x_feat, y_feat]].corr().iloc[0, 1]
    dr_q1 = samp[samp[x_feat] <= samp[x_feat].quantile(0.25)]['default_flag'].mean()
    dr_q4 = samp[samp[x_feat] >= samp[x_feat].quantile(0.75)]['default_flag'].mean()
    st.info(
        f"**{x_feat} vs {y_feat}** — Pearson r = {corr_val:.3f}  |  "
        f"Default rate in bottom quartile of {x_feat}: **{dr_q1:.1%}**  |  "
        f"Default rate in top quartile: **{dr_q4:.1%}**"
    )

# ── WOE / IV ─────────────────────────────────────────────────────────────────
elif page == "🔢 WoE / IV Analysis":
    st.title("🔢 Weight of Evidence & Information Value")

    if not ARTEFACTS_OK:
        st.error("Artefacts not found. Run the notebook to generate improved_pipe.pkl and woe_artefacts.pkl.")
        st.stop()

    def compute_woe_iv(df, feature, target='default_flag', n_bins=10, categorical=False):
        EPS=1e-6; te=df[target].sum(); tne=len(df)-te
        if categorical:
            groups=df.groupby(feature)[target].agg(['sum','count'])
        else:
            df2=df[[feature,target]].dropna().copy()
            df2['_bin']=pd.qcut(df2[feature],q=n_bins,duplicates='drop')
            groups=df2.groupby('_bin',observed=True)[target].agg(['sum','count'])
        groups.columns=['events','total']
        groups['nonevents']=groups['total']-groups['events']
        groups['pct_e']=groups['events']/(te+EPS)
        groups['pct_ne']=groups['nonevents']/(tne+EPS)
        groups['woe']=np.log((groups['pct_e']+EPS)/(groups['pct_ne']+EPS))
        groups['iv']=(groups['pct_e']-groups['pct_ne'])*groups['woe']
        groups['default_rate']=groups['events']/groups['total']
        return groups.reset_index()

    @st.cache_data
    def compute_all_ivs():
        train_raw = loan_df[loan_df['set']=='train'].copy().reset_index(drop=True)
        train = clean_dataframe(train_raw)
        num_cols = ['dti_ratio','credit_utilisation_pct','num_delinquencies_2yr','interest_rate',
                    'employment_length_years','annual_income','loan_amount','num_hard_inquiries_6mo',
                    'months_since_oldest_account','age','pct_accounts_current','total_revolving_balance',
                    'months_at_current_address','num_open_accounts']
        cat_cols = ['home_ownership','loan_purpose','email_domain_type']
        ivs = {}
        woe_dfs = {}
        for col in num_cols:
            wt = compute_woe_iv(train, col, n_bins=10)
            ivs[col] = wt['iv'].sum()
            woe_dfs[col] = wt
        for col in cat_cols:
            wt = compute_woe_iv(train, col, categorical=True)
            ivs[col] = wt['iv'].sum()
            woe_dfs[col] = wt
        return ivs, woe_dfs

    ivs, woe_dfs = compute_all_ivs()
    iv_df = pd.Series(ivs).sort_values(ascending=False).reset_index()
    iv_df.columns = ['feature','iv']
    iv_df['strength'] = iv_df['iv'].apply(lambda v: 'Strong' if v>=0.3 else 'Medium' if v>=0.1 else 'Weak')

    st.subheader("IV ranking — feature predictive power")
    colours = [TEAL if v>=0.3 else BLUE if v>=0.1 else CORAL for v in iv_df['iv']]
    fig = go.Figure(go.Bar(x=iv_df['iv'], y=iv_df['feature'], orientation='h',
                           marker_color=colours, text=iv_df['iv'].round(3),
                           textposition='outside'))
    fig.add_vline(x=0.1, line_dash='dash', line_color=AMBER, annotation_text='Medium (0.10)')
    fig.add_vline(x=0.3, line_dash='dash', line_color=CORAL, annotation_text='Strong (0.30)')
    fig.update_layout(**PLOT_LAYOUT, height=500, xaxis_title='Information Value')
    st.plotly_chart(fig, use_container_width=True)
    st.caption("IV < 0.02 unpredictive · 0.02–0.10 weak · 0.10–0.30 medium · 0.30–0.50 strong · > 0.50 very strong")

    st.subheader("WoE profile")
    woe_feat = st.selectbox("Feature", list(woe_dfs.keys()), index=0)
    wt = woe_dfs[woe_feat].copy()
    bin_col = wt.columns[0]
    wt['bin_str'] = wt[bin_col].astype(str)
    wt['colour'] = [CORAL if w>=0 else TEAL for w in wt['woe']]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(x=wt['woe'], y=wt['bin_str'], orientation='h',
                               marker_color=wt['colour'].tolist(),
                               text=wt['woe'].round(3), textposition='outside'))
        fig.add_vline(x=0, line_color=GRAY)
        fig.update_layout(**PLOT_LAYOUT, xaxis_title='WoE', title=f'{woe_feat} — WoE')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure(go.Bar(x=(wt['default_rate']*100).round(1), y=wt['bin_str'],
                               orientation='h', marker_color=AMBER,
                               text=(wt['default_rate']*100).round(1), texttemplate='%{text}%',
                               textposition='outside'))
        fig.update_layout(**PLOT_LAYOUT, xaxis_title='Default rate %', title=f'{woe_feat} — default rate per bin')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full WoE table")
    display_wt = wt[[bin_col,'events','nonevents','total','default_rate','woe','iv']].round(4)
    st.dataframe(display_wt, use_container_width=True)

# ── MODEL PERFORMANCE ─────────────────────────────────────────────────────────
elif page == "📉 Model Performance":
    st.title("📉 Model Performance")

    if not ARTEFACTS_OK:
        st.error("Artefacts not found. Run the notebook first.")
        st.stop()

    y_test, proba = get_test_predictions()
    auc = roc_auc_score(y_test, proba)
    gini = 2*auc - 1
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    ks = float(np.max(tpr - fpr))

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("AUC (improved LR)",  f"{auc:.4f}")
    c2.metric("Gini",               f"{gini:.4f}")
    c3.metric("KS statistic",       f"{ks:.4f}")
    c4.metric("Baseline AUC",       "0.772", delta=f"{auc-0.772:+.3f}")
    c5.metric("LightGBM ceiling",   "0.820", delta=f"{auc-0.820:+.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr[::15], y=tpr[::15], mode='lines', name=f'Improved LR (AUC={auc:.3f})',
                                 line=dict(color=TEAL, width=2.5)))
        # Approximate baseline ROC
        base_fpr = np.linspace(0,1,100)
        base_tpr = np.power(base_fpr, 1/(0.772*2.1))
        fig.add_trace(go.Scatter(x=base_fpr, y=base_tpr, mode='lines', name='Baseline LR (AUC=0.772)',
                                 line=dict(color=CORAL, width=1.8, dash='dash')))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random',
                                 line=dict(color=GRAY, dash='dot', width=1)))
        fig.update_layout(**PLOT_LAYOUT, xaxis_title='FPR', yaxis_title='TPR',
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Precision-Recall curve")
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec_arr[::15], y=prec_arr[::15], mode='lines',
                                 name='Improved LR', line=dict(color=TEAL, width=2.5)))
        fig.add_hline(y=float(y_test.mean()), line_dash='dot', line_color=GRAY,
                      annotation_text='No-skill baseline')
        fig.update_layout(**PLOT_LAYOUT, xaxis_title='Recall', yaxis_title='Precision',
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Confusion matrix at selected threshold")
    thresh = st.slider("Decision threshold", 0.05, 0.65, 0.25, 0.01)
    preds = (proba >= thresh).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("True approvals (TN)",  f"{tn:,}")
    c2.metric("Missed defaults (FP)", f"{fp:,}")
    c3.metric("Unfair declines (FN)", f"{fn:,}")
    c4.metric("Defaults caught (TP)", f"{tp:,}")

    fig = px.imshow([[tn,fp],[fn,tp]], text_auto=True,
                    x=['Pred: No Default','Pred: Default'],
                    y=['Act: No Default','Act: Default'],
                    color_continuous_scale=[[0,'#0D3D2A'],[1,'#3D1A10']])
    fig.update_layout(**PLOT_LAYOUT, height=280, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Coefficient analysis")
    coef_vals = improved_pipe.named_steps['lr'].coef_[0]
    feats = woe_artefacts['model_features']
    coef_df = pd.DataFrame({'feature':feats,'coef':coef_vals}).sort_values('coef')
    colours = [TEAL if v<0 else CORAL for v in coef_df['coef']]
    fig = go.Figure(go.Bar(x=coef_df['coef'], y=coef_df['feature'], orientation='h',
                           marker_color=colours,
                           text=coef_df['coef'].round(3), textposition='outside'))
    fig.add_vline(x=0, line_color=GRAY)
    fig.update_layout(**PLOT_LAYOUT, height=600,
                      xaxis_title='Standardised coefficient (log-odds per SD)',
                      title='Logistic Regression Coefficients (teal=protective, coral=risk-increasing)')
    st.plotly_chart(fig, use_container_width=True)

# ── BUSINESS DASHBOARD ────────────────────────────────────────────────────────
elif page == "💼 Business Dashboard":
    st.title("💼 Business Decision Dashboard")

    if not ARTEFACTS_OK:
        st.error("Artefacts not found. Run the notebook first.")
        st.stop()

    y_test, proba = get_test_predictions()
    N = len(y_test)

    st.subheader("Scenario assumptions")
    sa1, sa2, sa3 = st.columns(3)
    with sa1:
        avg_loan = st.slider("Avg loan size (R)", 2000, 30000, 10000, 500,
                             help="Average outstanding loan balance")
    with sa2:
        revenue_rate = st.slider("Annual revenue rate %", 3.0, 20.0, 8.0, 0.5,
                                 help="Interest + fee income as % of loan per year") / 100
    with sa3:
        lgd = st.slider("Loss Given Default (LGD) %", 10, 90, 55, 5,
                        help="% of loan balance lost when a borrower defaults") / 100
    st.caption(
        f"P&L formula: (approved × R{avg_loan:,} × {revenue_rate:.0%} revenue) "
        f"− (defaults in approved × R{avg_loan:,} × {lgd:.0%} LGD)"
    )

    st.subheader("Threshold sensitivity analysis")
    thresholds = np.arange(0.05, 0.65, 0.01)
    rows = []
    for t in thresholds:
        approved = proba < t
        n_app = approved.sum()
        if n_app == 0: continue
        def_in_app = y_test[approved].sum()
        def_caught = y_test[~approved].sum()
        net_pl = (n_app * avg_loan * revenue_rate - def_in_app * avg_loan * lgd) / 1e6
        rows.append({'threshold':round(t,2),
                     'approval_rate':n_app/N,
                     'book_default_rate':def_in_app/n_app,
                     'recall':def_caught/y_test.sum(),
                     'false_declines':int((~approved & (y_test==0)).sum()),
                     'net_pl':net_pl})
    thresh_df = pd.DataFrame(rows)
    opt_t = thresh_df.loc[thresh_df['net_pl'].idxmax(), 'threshold']

    col1, col2 = st.columns([1,2])
    with col1:
        t = st.slider("Approval threshold", 0.05, 0.64, 0.25, 0.01,
                      help="Approve applications with predicted PD below this value")
        row = thresh_df[thresh_df['threshold']==round(t,2)].iloc[0]
        st.metric("Approval rate",      f"{row['approval_rate']:.1%}")
        st.metric("Book default rate",  f"{row['book_default_rate']:.1%}")
        st.metric("Defaults caught",    f"{row['recall']:.1%}")
        st.metric("Good applicants denied", f"{int(row['false_declines']):,}")
        st.metric("Est. net P&L",       f"R{row['net_pl']:.2f}M")
        if abs(t - opt_t) < 0.015:
            st.success(f"★ Near profit-optimal threshold ({opt_t:.0%})")

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresh_df['threshold']*100, y=thresh_df['approval_rate']*100,
                                 name='Approval rate %', line=dict(color=TEAL, width=2)))
        fig.add_trace(go.Scatter(x=thresh_df['threshold']*100, y=thresh_df['book_default_rate']*100,
                                 name='Book default rate %', line=dict(color=CORAL, width=2, dash='dash'),
                                 yaxis='y2'))
        fig.add_vline(x=t*100, line_color=AMBER, line_dash='dash', annotation_text=f'Current: {t:.0%}')
        fig.update_layout(**PLOT_LAYOUT, yaxis_title='Approval rate %',
                          yaxis2=dict(title='Book default rate %', overlaying='y', side='right',
                                      gridcolor='#2E312D', color=CORAL),
                          legend=dict(bgcolor='rgba(0,0,0,0)'),
                          title='Volume vs Risk trade-off')
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estimated net P&L vs threshold")
        fig = go.Figure(go.Scatter(x=thresh_df['threshold']*100, y=thresh_df['net_pl'],
                                   fill='tozeroy', line=dict(color=TEAL, width=2.5),
                                   fillcolor='rgba(29,158,117,0.1)'))
        fig.add_vline(x=opt_t*100, line_color=AMBER, line_dash='dash',
                      annotation_text=f'Optimal: {opt_t:.0%} → R{thresh_df.net_pl.max():.2f}M')
        fig.add_vline(x=t*100, line_color=GRAY, line_dash='dot')
        fig.update_layout(**PLOT_LAYOUT, xaxis_title='Threshold %', yaxis_title='Net P&L (ZAR M)',
                          title=f'P&L: R{avg_loan:,} avg loan · {revenue_rate:.0%} revenue · {lgd:.0%} LGD')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Decision band breakdown")
        band_order = ['Auto-approve','Approve','Manual review','Decline','Hard decline']
        band_cuts  = [0, 0.08, 0.18, 0.30, 0.50, 1.0]
        band_labels = band_order
        bands = pd.cut(proba, bins=band_cuts, labels=band_labels, include_lowest=True)
        band_df = pd.DataFrame({'band':bands,'actual':y_test})
        bsummary = band_df.groupby('band', observed=False).agg(n=('actual','count'), dr=('actual','mean')).reset_index()
        bsummary['pct'] = bsummary['n']/N
        band_colours = {'Auto-approve':TEAL,'Approve':'#5DCAA5','Manual review':AMBER,'Decline':CORAL,'Hard decline':'#8B2000'}
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Volume %', x=bsummary['band'], y=bsummary['pct']*100,
                             marker_color=[band_colours[b] for b in bsummary['band']], yaxis='y'))
        fig.add_trace(go.Scatter(name='Actual default rate', x=bsummary['band'],
                                 y=bsummary['dr']*100, mode='lines+markers',
                                 line=dict(color='white',width=2), marker=dict(size=8), yaxis='y2'))
        fig.update_layout(**PLOT_LAYOUT, yaxis_title='% of applicants',
                          yaxis2=dict(title='Default rate %', overlaying='y', side='right',
                                      gridcolor='#2E312D', color='white'),
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Policy framework")
    policy_data = {
        'PD band':       ['< 8%',    '8–18%',           '18–30%',          '30–50%',  '> 50%'],
        'Decision':      ['Auto-approve','Approve',      'Manual review',   'Decline', 'Hard decline'],
        'Rate / action': ['Standard', '+1–2% risk adj.','Request docs',    'Adverse action notice','No review 12mo'],
        'Regulatory':    ['Low risk', 'Manageable',      'Borderline',      'ECOA/FCA','ECOA/FCA'],
    }
    st.dataframe(pd.DataFrame(policy_data), use_container_width=True, hide_index=True)

# ── CREDIT SCORER ─────────────────────────────────────────────────────────────
elif page == "🧾 Credit Scorer":
    st.title("🧾 Live Credit Scorer")
    st.caption("Instant default probability from the trained logistic regression model.")

    if not ARTEFACTS_OK:
        st.error("Artefacts not found. Run the notebook first.")
        st.stop()

    with st.form("scorer_form"):
        st.subheader("Applicant details")
        c1, c2, c3 = st.columns(3)
        with c1:
            age             = st.slider("Age",                       18, 75,  32)
            annual_income   = st.slider("Annual income (£)",       8000, 200000, 45000, step=1000)
            loan_amount     = st.slider("Loan amount (£)",          500, 40000,  10000, step=500)
            employment_yrs  = st.slider("Employment years",           0, 35,     5)
        with c2:
            dti             = st.slider("DTI ratio",               0.05, 0.60,  0.25)
            credit_util     = st.slider("Credit utilisation %",       0, 86,    35)
            interest_rate   = st.slider("Interest rate %",          5.0, 27.0,  12.0, step=0.1)
            months_oldest   = st.slider("Months since oldest account", 0, 300,  72)
        with c3:
            delinquencies   = st.slider("Delinquencies (2yr)",        0, 9,     0)
            hard_inquiries  = st.slider("Hard inquiries (6mo)",       0, 10,    1)
            pct_current     = st.slider("Accounts current %",        50, 100,   95)
            num_accounts    = st.slider("Number of accounts",         1, 20,    5)
            home_ownership  = st.selectbox("Home ownership", ['RENT','MORTGAGE','OWN','OTHER'])
            loan_purpose    = st.selectbox("Loan purpose",  ['debt_consolidation','home_improvement','major_purchase','medical','education','other','small_business'])
            ever_delinquent = st.selectbox("Ever delinquent?", [0,1], format_func=lambda x: 'Yes' if x else 'No')

        submitted = st.form_submit_button("Score applicant", use_container_width=True)

    if submitted:
        # Build single-row dataframe matching the full feature set
        caps = train_stats['caps']
        row = {
            'age':                    age,
            'annual_income':          min(annual_income, caps['annual_income']),
            'loan_amount':            min(loan_amount,   caps['loan_amount']),
            'employment_length_years': employment_yrs,
            'dti_ratio':              dti,
            'credit_utilisation_pct': min(credit_util, caps['credit_utilisation_pct']),
            'interest_rate':          interest_rate,
            'months_since_oldest_account': months_oldest,
            'num_delinquencies_2yr':  delinquencies,
            'num_hard_inquiries_6mo': hard_inquiries,
            'pct_accounts_current':   pct_current,
            'num_open_accounts':      num_accounts,
            'total_revolving_balance': 5000,    # typical population median
            'months_at_current_address': 36,
            'home_ownership':         home_ownership,
            'loan_purpose':           loan_purpose,
            'email_domain_type':      'free',
            'ever_delinquent':        ever_delinquent,
            'annual_income_missing':  0,
            'employment_length_years_missing': 0,
            'num_open_accounts_missing': 0,
        }
        input_df = pd.DataFrame([row])

        # Engineer features
        input_fe = build_features(input_df, woe_artefacts)
        X_input  = input_fe[woe_artefacts['model_features']]
        pd_score = float(improved_pipe.predict_proba(X_input)[0, 1])

        # Scorecard points
        PDO=20; OFFSET=600
        factor = PDO / np.log(2)
        coefs_raw   = improved_pipe.named_steps['lr'].coef_[0]
        scaler_obj  = improved_pipe.named_steps['sc']
        coefs_uns   = coefs_raw / scaler_obj.scale_
        feature_contribs = dict(zip(woe_artefacts['model_features'], -coefs_uns * factor))

        score_total = OFFSET + sum(
            feature_contribs[f] * float(X_input[f].iloc[0])
            for f in woe_artefacts['model_features']
        )

        # Output
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Probability of default", f"{pd_score:.1%}")
        c2.metric("Scorecard points",       f"{score_total:.0f}")

        if   pd_score < 0.08: decision, colour = "✅ Auto-approve — standard rate",  TEAL
        elif pd_score < 0.18: decision, colour = "✅ Approve — risk-adjusted rate",   TEAL
        elif pd_score < 0.30: decision, colour = "⚠️ Manual review required",         AMBER
        elif pd_score < 0.50: decision, colour = "❌ Decline",                         CORAL
        else:                 decision, colour = "❌ Hard decline",                    CORAL
        c3.metric("Decision", decision)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pd_score*100,
            title={'text':'Default probability %','font':{'color':'#9BA396','size':13}},
            number={'suffix':'%','font':{'color':'#E8EBE4','size':28}},
            gauge={
                'axis':{'range':[0,100],'tickcolor':'#636660'},
                'bar':{'color':colour,'thickness':0.3},
                'bgcolor':'#2E312D',
                'steps':[
                    {'range':[0,8],   'color':'#0D3D2A'},
                    {'range':[8,18],  'color':'#133D2A'},
                    {'range':[18,30], 'color':'#3D2A06'},
                    {'range':[30,50], 'color':'#3D1A10'},
                    {'range':[50,100],'color':'#2B0D07'},
                ],
            }
        ))
        fig.update_layout(paper_bgcolor='#161714', font=dict(color='#9BA396'), height=260,
                          margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Top adverse reasons
        st.subheader("Top adverse action reasons")
        contrib_df = pd.DataFrame({
            'feature': woe_artefacts['model_features'],
            'value':   [float(X_input[f].iloc[0]) for f in woe_artefacts['model_features']],
            'coef':    coefs_raw,
        })
        contrib_df['contribution'] = contrib_df['coef'] * contrib_df['value']
        top_risk = contrib_df.sort_values('contribution', ascending=False).head(5)
        top_prot = contrib_df.sort_values('contribution').head(5)

        col1, col2 = st.columns(2)
        with col1:
            st.caption("Top risk-increasing factors")
            for _, r in top_risk.iterrows():
                st.markdown(f"🔴 **{r['feature'].replace('_',' ')}** (+{r['contribution']:.3f})")
        with col2:
            st.caption("Top protective factors")
            for _, r in top_prot.iterrows():
                st.markdown(f"🟢 **{r['feature'].replace('_',' ')}** ({r['contribution']:.3f})")