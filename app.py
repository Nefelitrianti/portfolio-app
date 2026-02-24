import streamlit as st
from streamlit.components.v1 import iframe
from streamlit_option_menu import option_menu
from pathlib import Path


if "project_opened" not in st.session_state:
    st.session_state.project_opened = False
if "selected_project" not in st.session_state:
    st.session_state.selected_project = None

APP_DIR = Path(__file__).parent


st.set_page_config(page_title="Portfolio-App", layout="wide")


st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none !important;}
</style>
""", unsafe_allow_html=True)


selected = option_menu(
    menu_title=None,
    options=["Portfolio", "Projects"],
    icons=["person", "file-earmark-text"],  
    orientation="horizontal",
    default_index=0,
)


if selected == "Portfolio":
    st.title("Data & Analytics Portfolio")

    st.markdown(
        """
        <style>
        .intro-text {
            font-size: 1.05rem;
            line-height: 1.8;
            color: #90CAF9;
            margin-bottom: 10px;
        }
        .intro-text strong {
            color: #42A5F5;
            font-size: 1.08rem;
        }
        .skill-header {
            font-size: 1.15rem;
            font-weight: 700;
            color: #1E88E5;
            margin-top: 16px;
            margin-bottom: 4px;
        }
        .skill-item {
            font-size: 1rem;
            color: #90CAF9;
            margin-left: 16px;
            line-height: 1.8;
        }
        .skill-item strong { color: #42A5F5; }
        hr.blue-divider {
            border: none;
            border-top: 1px solid #1E88E5;
            margin: 16px 0;
        }
        </style>

        <p class="intro-text">
        I bridge the gap between complex statistical theory and automated financial systems. 
        My work focuses on identifying the underlying drivers of <strong>European economic shifts</strong> 
        by synthesizing demographic data with real-time market performance.
        </p>

        <div class="skill-header">📊 Statistics (Python & R)</div>
        <div class="skill-item">• Implementing Bayesian Factor Analysis to extract latent variables from University of Bocconi datasets.</div>
        <div class="skill-item">• Utilizing Bayes Factors for rigorous model selection and hypothesis testing in demographic research.</div>

        <div class="skill-header">⚙️ Automated Data Pipelines (SQL & Python)</div>
        <div class="skill-item">• Engineered an automated ETL system to scrape and process Yahoo Finance data for a 6-stock portfolio.</div>
        <div class="skill-item">• Managed high-frequency closing prices via a structured MySQL database.</div>

        <div class="skill-header">📈 Interactive Visualization (RShiny & Power BI)</div>
        <div class="skill-item">• Developed RShiny dashboards to visualize Official European Statistics and demographic trends.</div>
        <div class="skill-item">• Leveraged Power BI for deep-dive relationship analysis between stock volatility and external economic factors.</div>
        """,
        unsafe_allow_html=True,
    )

    st.html(
        """
        <style>
        .custom-expander {
            border-radius: 8px;
            margin-bottom: 12px;
            overflow: hidden;
        }
        .custom-expander details { padding: 0; }
        .custom-expander summary {
            padding: 14px 18px;
            font-size: 1.15rem;
            font-weight: 700;
            cursor: pointer;
            list-style: none;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .custom-expander summary::-webkit-details-marker { display: none; }
        .custom-expander summary::after {
            content: '▸';
            margin-left: auto;
            transition: transform 0.2s;
        }
        .custom-expander details[open] summary::after { transform: rotate(90deg); }
        .custom-expander .content { padding: 16px 18px; font-size: 0.97rem; line-height: 1.7; }

        .exp1 { border-left: 4px solid #1565C0; background: rgba(21, 101, 192, 0.08); }
        .exp1 summary { color: #1565C0; }
        .exp2 { border-left: 4px solid #1E88E5; background: rgba(30, 136, 229, 0.08); }
        .exp2 summary { color: #1E88E5; }
        .exp3 { border-left: 4px solid #42A5F5; background: rgba(66, 165, 245, 0.08); }
        .exp3 summary { color: #42A5F5; }
        .exp4 { border-left: 4px solid #90CAF9; background: rgba(144, 202, 249, 0.08); }
        .exp4 summary { color: #90CAF9; }
        </style>

        <div class="custom-expander exp1">
          <details open>
            <summary> Global Social Impact: Bayesian Factor Analysis</summary>
            <div class="content">
              <b>The Problem:</b> How do demographics and work environments influence social and psychological outcomes?<br><br>
              <b>The Analysis:</b><br>
              • <b>Data Engineering:</b> Standardized Likert-scale survey data from Bocconi University.<br>
              • <b>Feature Creation:</b> Calculated 'Remote Ratios' to categorize workers into Hybrid, Remote, or Office cohorts.<br>
              • <b>Modeling:</b> Reduced high-dimensional data into 4 Core Factors: Flexibility, Challenges, Career Anxiety, and WLB Struggle.<br>
              • <b>Statistical Rigor:</b> Applied Bayes Factors (BF) to prove significant differences across ethnicity, age, and gender.
            </div>
          </details>
        </div>

        <div class="custom-expander exp2">
          <details>
            <summary> SES Earnings: European Inequality Explorer</summary>
            <div class="content">
              <b>The Problem:</b> Does high nominal income translate to real-world purchasing power, and where do gender opportunity gaps persist?<br><br>
              <b>The Analysis:</b><br>
              • <b>Integration:</b> Connected R and RShiny via the ses_hour library.<br>
              • <b>Economics:</b> Compared Nominal Wages vs. Purchasing Power Standards (PPS) across Europe.<br>
              • <b>Insight:</b> Extracted occupational data to reveal Horizontal Segregation—identifying sectors where women are underrepresented.
            </div>
          </details>
        </div>

        <div class="custom-expander exp3">
          <details>
            <summary> Italy Demographics Data</summary>
            <div class="content">
              <b>The Focus:</b> Analyzing fertility, mortality and population structure across the years.<br><br>
              <b>The Analysis:</b><br>
              • Evaluated the fertility and mortality rate based on social and macroeconomic factors.
            </div>
          </details>
        </div>

        <div class="custom-expander exp4">
          <details>
            <summary>ROI Analysis for 6-Stock Portfolio</summary>
            <div class="content">
              <b>The Objective:</b> Analyzing how portfolio diversification, market conditions, and external costs impact long-term ROI.<br><br>
              <b>The Data Pipeline:</b><br>
              • <b>Automated Data Retrieval:</b> Developed a pipeline to fetch daily closing prices via Yahoo Finance API.<br>
              • <b>Database Management:</b> Engineered an ETL process to store real-time financial data in a MySQL database.<br><br>
              <b>The Analysis & Visualization:</b><br>
              • <b>Power BI Integration:</b> Leveraged Power BI for complex table relationships and interactive visualizations.<br>
              • <b>Key Insights:</b> Dashboard illustrates compounding effects of inflation, brokerage fees, and diversification on net returns.
            </div>
          </details>
        </div>
        """
    )


elif selected == "Projects":
    PROJECTS = {
        "Factor Analysis": {
            "type": "python_script",
            "summary": "Statistical factor analysis using Python. Explores underlying variables.",
            "script_path": "scripts/untitled16.py",
        },
        "SES Earnings App": {
            "type": "shiny",
            "summary": "EU Structure of Earnings Survey explorer.",
            "url": "https://nefelitrianti.shinyapps.io/ses-earnings/",
        },
        "Italy Data 1960-2020": {
            "type": "shiny",
            "summary": "Demographic Data Italy 1960-2020.",
            "url": "https://nefelitrianti.shinyapps.io/data_italy/",
        },
        "Investment Dashboard": {
            "type": "powerbi",
            "summary": "KPIs and performance analytics.",
            "image_path": "assets/powerbi_dashboard.png",
        },
    }

    
    if not st.session_state.project_opened:
        st.title("Projects")

        name = st.selectbox(
            "Select a project",
            list(PROJECTS.keys()),
            index=None,
            placeholder="Select a project...",
            label_visibility="collapsed",
        )

        if name:
            st.info(PROJECTS[name]["summary"])
            st.session_state.selected_project = name
            st.session_state.project_opened = True
            st.rerun()

    
    else:
        if st.button("⬅️ Back to Projects"):
            st.session_state.project_opened = False
            st.session_state.selected_project = None
            st.rerun()

        selected_name = st.session_state.selected_project
        if not selected_name or selected_name not in PROJECTS:
            st.warning("No project selected. Go back and select a project.")
            st.session_state.project_opened = False
            st.session_state.selected_project = None
            st.rerun()

        p = PROJECTS[selected_name]

        
        if p["type"] == "python_script":
            script_file = APP_DIR / p["script_path"]
            if script_file.exists():
                with open(script_file, "r", encoding="utf-8") as f:
                    code = f.read()

                
                st.header(selected_name)

                
                scope = {"st": st}

                try:
                    exec(code, scope, scope)
                except Exception as e:
                    st.error(f"Error running script: {e}")
            else:
                st.error(f"Script file not found: {p['script_path']}")

        
        elif p["type"] == "powerbi":
            st.header(selected_name)
            img = APP_DIR / p["image_path"]
            if img.exists():
                st.image(img, use_container_width=True)
            else:
                st.error(f"Image not found: {p['image_path']}")

        
        elif p["type"] == "shiny":
            st.header(selected_name)
            iframe(p["url"], height=800, scrolling=True)
            st.link_button("Open App in New Tab", p["url"])
