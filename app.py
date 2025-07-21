import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import altair as alt

st.set_page_config(
    page_title="Calculateur de Survie Médiane",
    page_icon="🩺"  # Tu peux aussi mettre "demo/logo.png" si tu veux utiliser ton logo
)

# --- Personnalisation du thème Streamlit (CSS) ---
st.markdown(
    """
    <style>
    body, .stApp, .main, .block-container {
        background-color: #fff !important;
        color: #111 !important;
    }
    .title-main {color: #2563eb; font-size: 2.8em; font-weight: bold; letter-spacing: 1px;}
    .subtitle {font-size: 1.2em; margin-bottom: 1.5em; color: #111 !important;}
    .form-label, label, .stSlider label, .stNumberInput label, .stSelectbox label {
        color: #111 !important;
    }
    .result-card {border-radius: 12px; padding: 1.5em; margin-top: 1.5em; box-shadow: 0 2px 8px #b6c6e2; color: #111 !important;}
    .footer {color: #6b7280; font-size: 0.95em; margin-top: 2em; text-align: center;}
    .logo {height: 60px; margin-bottom: 1em;}
    .sidebar-content label {
        color: #fff !important;
    }
    .stFormSubmitButton > button {
        color: #fff !important;
        background-color: #111 !important;
        font-weight: bold;
        border-radius: 8px;
        border: 1px solid #111;
        transition: background 0.2s;
    }
    .stFormSubmitButton > button:hover {
        background-color: #2563eb !important;
        color: #fff !important;
    }
    .stRadio > p {
        coloc: #fff
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar navigation ---
menu = st.sidebar.radio(
    "Navigation",
    ("Accueil", "Prédiction", "Méthodologie", "À propos")
)

# --- Fonction utilitaire pour afficher le logo ---
def afficher_logo():
    st.image("Logo EPSILON.png", width=200, output_format="PNG", use_container_width=False)

# --- Présentation de l'entreprise ---
def presentation_entreprise():
    st.markdown("""
    <div style='border-radius:10px; padding:1em 1.5em; margin-bottom:1.5em;'>
    <b>À propos d'Epsilon :</b><br>
    Epsilon est une entreprise innovante spécialisée dans l'analyse de données et le développement de solutions numériques sur mesure pour le secteur de la santé. Grâce à son expertise en intelligence artificielle et en data science, Epsilon accompagne les professionnels et établissements de santé dans l'optimisation de leurs processus, la valorisation de leurs données et l'amélioration de la prise en charge des patients.<br>
    </div>
    """, unsafe_allow_html=True)

# --- Charger modèle, scaler et encodeur (pour la page Prédiction) ---
@st.cache_resource
def load_model():
    return joblib.load('weibull_aft_model.pkl')
@st.cache_resource
def load_scaler():
    return joblib.load('standard_scaler.pkl')
@st.cache_resource
def load_encoder():
    return joblib.load('tumor_ordinal_encoder.pkl')

# --- Page Accueil ---
if menu == "Accueil":
    afficher_logo()
    st.markdown('<div class="title-main">Analyse de Survie Médiane</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Bienvenue sur le calculateur de durée médiane de survie pour le cancer colorectal.</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='border-radius:10px; padding:1em 1.5em; margin-bottom:2em'>
    <b>Objectif :</b> Ce projet vise à fournir un outil interactif d'estimation de la survie médiane, basé sur des modèles statistiques avancés et des données cliniques réelles.<br><br>
    <b>Fonctionnalités principales :</b>
    <ul>
    <li>Prédiction personnalisée de la survie médiane</li>
    <li>Explications sur la méthodologie et les données</li>
    <li>Interface moderne et pédagogique</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    #st.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80", caption="Image illustrative - Unsplash", use_container_width=True)
    st.image("Patient âgé.png", caption="", use_container_width=True)

# --- Page Prédiction ---
elif menu == "Prédiction":
    afficher_logo()
    st.markdown('<div class="title-main">Calculateur de Survie Médiane</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Estimez la durée médiane de survie selon vos paramètres cliniques.</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='border-radius:10px; padding:1em 1.5em; margin-bottom:2em;'>
    <b>À propos :</b> Ce calculateur utilise un modèle de survie avancé (Weibull AFT) pour estimer la durée médiane de survie à partir de vos données. Les résultats sont donnés à titre informatif et ne remplacent pas un avis médical.
    </div>
    """, unsafe_allow_html=True)
    model = load_model()
    scaler = load_scaler()
    encoder = load_encoder()
    st.markdown('<h4 style="color:#2563eb;">Paramètres cliniques du patient</h4>', unsafe_allow_html=True)
    with st.form("formulaire_survie"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Âge du patient", 38, 90, 60, help="Âge en années révolues.")
            poids = st.number_input("Poids (kg)", min_value=30.0, max_value=200.0, value=70.0, format="%.1f", step=0.1, help="Poids du patient en kilogrammes.")
        with col2:
            tumor_stage = st.selectbox(
                "Stade tumoral (classification AJCC)",
                [1, 2, 3, 4],
                format_func=lambda x: {1: "I", 2: "II", 3: "III", 4: "IV"}[x],
                help="Stade d'évolution de la tumeur : I (précoce) à IV (avancé)"
            )
            taille = st.number_input("Taille (m)", min_value=1.0, max_value=2.5, value=1.70, format="%.2f", help="Taille du patient en mètres.")
        submit = st.form_submit_button("Prédire la durée médiane de survie", use_container_width=True)
    if submit:
        tumor_stage_label = ['I', 'II', 'III', 'IV'][tumor_stage-1]
        if tumor_stage_label in ['I', 'II']:
            tumor_stage_label = 'I-II'
        tumor_stage_encoded = encoder.transform([[tumor_stage_label]])[0,0]
        X_num = scaler.transform([[age, poids / (taille ** 2)]])[0]
        age_std, BMI_std = X_num[0], X_num[1]
        age_squared = age_std ** 2
        age_cubed = age_std ** 3
        BMI_squared = BMI_std ** 2
        input_df = pd.DataFrame([{
            "age": age_std,
            "BMI": BMI_std,
            "tumor_stage": tumor_stage_encoded,
            "BMI_squared": BMI_squared,
            "age_squared": age_squared,
            "age_cubed": age_cubed
        }])
        try:
            prediction = model.predict_median(input_df)
            valeur = prediction.values[0]
            survival_function = model.predict_survival_function(input_df)[0]
            # Affichage de la courbe de survie avec fond blanc
            surv_df = survival_function.reset_index()
            surv_df.columns = ['Mois', 'Survie']
            chart = alt.Chart(surv_df).mark_line(color='#2563eb', strokeWidth=3).encode(
                x=alt.X('Mois', title='Temps (mois)', axis=alt.Axis(labelFontSize=16, titleFontSize=18, labelColor='#111', titleColor='#111')),
                y=alt.Y('Survie', title='Probabilité de survie', scale=alt.Scale(domain=[0,1]), axis=alt.Axis(labelFontSize=16, titleFontSize=18, labelColor='#111', titleColor='#111')),
            ).properties(
                width=600,
                height=350,
                background='#fff'
            )
            st.altair_chart(chart, use_container_width=True)
            # Calcul du temps pour survie < 20%
            import numpy as np
            t_grid = np.linspace(0, 400, 2000)
            surv = model.predict_survival_function(input_df, times=t_grid)
            t_5 = None
            for t, s in zip(t_grid, surv.values.T[0]):
                if s <= 0.25:
                    t_5 = t
                    break
            if pd.isna(valeur) or valeur == float('inf') or valeur == float('-inf'):
                st.error("La prédiction retournée est invalide (infinie ou non numérique). Veuillez vérifier les valeurs saisies ou contacter l'administrateur du modèle.")
            else:
                if t_5 is not None:
                    t5_str = f"{t_5:.2f} mois ~ {t_5/12:.2f} ans"
                else:
                    t5_str = "Non atteint dans l'intervalle (0-400 mois)"
                st.markdown(f"""
                <div class='result-card'>
                    <span style='font-size:1.3em; color:#2563eb; font-weight:bold;'>Durée médiane de survie estimée :</span><br>
                    <span style='font-size:2.2em; color:#111827; font-weight:bold;'>{valeur:.2f} mois ~ {valeur/12:.2f} ans</span><br><br>
                    <span style='font-size:1.1em; color:#374151;'>Durée pour une probabilité de survie de 25% :</span><br>
                    <span style='font-size:1.5em; color:#b91c1c; font-weight:bold;'>{t5_str}</span>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Erreur : {e}")

# --- Page Méthodologie ---
elif menu == "Méthodologie":
    afficher_logo()
    st.markdown('<div class="title-main">Méthodologie</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Comment fonctionne le modèle et quelles données sont utilisées ?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='border-radius:10px; padding:1em 1.5em; margin-bottom:2em;'>
    <b>Pipeline de traitement :</b>
    <ol>
    <li>Nettoyage et sélection des variables cliniques les plus pertinentes</li>
    <li>Imputation des valeurs manquantes (KNNImputer)</li>
    <li>Standardisation des variables numériques (StandardScaler)</li>
    <li>Encodage ordinal du stade tumoral</li>
    <li>Création de variables polynomiales (carré, cube)</li>
    <li>Entraînement d’un modèle Weibull AFT (lifelines)</li>
    </ol>
    <b>Données utilisées :</b> <br>
    - Données cliniques anonymisées de patients atteints de cancer colorectal.<br>
    - Variables : âge, stade tumoral, poids et taille.<br>
    </p>
    <b>Limites :</b> <br>
    - Prédiction indicative, dépendante de la qualité des données et du modèle<br>
    - Ne remplace pas un avis médical
    </div>
    """, unsafe_allow_html=True)

# --- Page À propos ---
elif menu == "À propos":
    afficher_logo()
    st.markdown('<div class="title-main">À propos</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Informations</div>', unsafe_allow_html=True)
    presentation_entreprise()
    st.markdown("""
    <div style='border-radius:10px; padding:1em 1.5em; margin-bottom:2em;'>
    <b>Développement :</b> Projet de soutenance Bachelor - Keyce Informatique<br>
    <b>Chef de projet :</b> Morryl KOUEMO<br>
    </div>
    """, unsafe_allow_html=True)

# --- Footer global ---
st.markdown("""
<div class='footer'>
Modèle de prédiction - <b>Analyse de survie</b>.<br>
</div>
""", unsafe_allow_html=True)
