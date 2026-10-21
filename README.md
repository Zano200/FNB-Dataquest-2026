Credit Risk Modelling System

FNB DataQuest 2026 · Team of 3

A credit risk model that predicts which borrowers in a loan book are likely to default, delivered through an interactive Streamlit dashboard so credit teams can explore risk without touching code.

The problem

Banks need to estimate the probability that a borrower will default before lending to them. A model that ranks risk well lets a lender approve more good customers while avoiding costly losses. Our task was to build and present a credit risk model on the provided loan book.

Approach
Data cleaning and exploratory analysis of the loan book (loan_book.csv)
Weight of Evidence (WoE) and Information Value (IV) encoding, the standard technique in bank credit scoring, to transform variables and select the features most predictive of default
Model training and evaluation using AUC (how well the model separates defaulters from non-defaulters)
Streamlit dashboard with 6 modules to present data, model results and risk insights
Results
Model	AUC
Baseline	0.680
Final model (with WoE/IV feature engineering)	0.795

Feature engineering lifted AUC by 0.115, a large improvement in the model's ability to rank borrower risk.

What's in this repo
File	What it is
credit_modelling_notebook (1).ipynb	Full analysis and modelling notebook
App_new.py	Streamlit dashboard app
woe_artefacts.pkl	Saved WoE/IV encoding artefacts used by the app
train_stats.pkl	Saved training statistics used by the app
loan_book.csv	Dataset
FNB CREDIT RISK MODELLING REPORT.docx	Final written report
FNB Powerpoint .pptx	Final presentation
student_task.pdf	Challenge brief
DataQuest26 Final.zip	Full submission package
Run the dashboard
bash
pip install streamlit pandas numpy scikit-learn matplotlib
streamlit run App_new.py


Tools

Python · Pandas · NumPy · Scikit-learn · WoE/IV encoding · Matplotlib · Streamlit · Jupyter
