Source
__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
* RMAP factor analysis dataset: RMAP_Data_Descriptor_Data.xlsx (Likert → numeric, EFA / factor scores)

* EU SES wage analysis (2010–2022): Eurostat Structure of Earnings Survey (SES) tables (download/API → SQL)

* Italy long-run dataset (1954–2020): historical macro time-series (cleaning → time-series analysis)

* 6-stocks portfolio dashboard: price data + inflation & fees adjustment (Power BI image/dashboard embedded in app)
__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
Features
*Portfolio Sections (Projects)

*Bayesian / Factor Analysis Dashboard

*Likert transformation

*Exploratory Factor Analysis (EFA)

*Factor score visualization & comparisons

__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

Usage

Clone the repository

git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

Install requirements

pip install -r requirements.txt

Run the Streamlit app

streamlit run app.py
Project Files

app.py → main Streamlit web application

pages/ → portfolio project pages (RMAP / SES / Italy / Finance)

data/ → place datasets here (or keep a .gitignore if large)

assets/ → images (Power BI screenshots, icons, etc.)

requirements.txt → dependencies

______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

Screenshots

Portfolio Home

<img width="1601" height="806" alt="Στιγμιότυπο οθόνης 2026-02-26 100559" src="https://github.com/user-attachments/assets/2ddfaf39-c4d0-4f11-83f5-71c9ab048f94" />


Projects Home 

<img width="1625" height="845" alt="image" src="https://github.com/user-attachments/assets/b4d83490-eac7-4321-b59d-96c6acc79a16" />


Bayesian Factor Analysis (RMAP)

(add image here)

SES Wage Analysis (EU 2010–2022)

(add image here)

Italy Long-run Economic Data (1954–2020)

(add image here)

Financial Portfolio Estimation (6 Stocks)

(add image here)

Αν θες, το κάνω 100% έτοιμο με σωστά markdown για screenshots, π.χ.:
