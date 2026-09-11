import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import shap
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="MedRisk — Prediction Readmission",
    page_icon="hospital",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.risk-critique { background-color: #fee2e2; border-left: 5px solid #dc2626; padding: 10px; border-radius: 4px; }
.risk-eleve    { background-color: #ffedd5; border-left: 5px solid #f97316; padding: 10px; border-radius: 4px; }
.risk-modere   { background-color: #fef9c3; border-left: 5px solid #eab308; padding: 10px; border-radius: 4px; }
.risk-faible   { background-color: #dcfce7; border-left: 5px solid #22c55e; padding: 10px; border-radius: 4px; }
.metric-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ── Chargement du modele ─────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("medrisk_model.pkl")
    features = joblib.load("medrisk_features.pkl")
    with open("medrisk_config.json") as f:
        config = json.load(f)
    return model, features, config

try:
    model, feature_names, config = load_model()
    THRESHOLD = config["threshold"]
    MODEL_LOADED = True
except Exception as e:
    st.error(f"Impossible de charger le modele : {e}")
    st.info("Lance d'abord le notebook MedRisk pour generer les fichiers .pkl et .json")
    MODEL_LOADED = False
    st.stop()

# ── Preprocessing ─────────────────────────────────────────
def preprocess(data_dict):
    data = pd.DataFrame([data_dict])
    le = LabelEncoder()
    for col in data.select_dtypes(include="object").columns:
        data[col] = le.fit_transform(data[col].astype(str))
    data["score_complexite"] = (
        data["num_lab_procedures"] / 132 +
        data["num_medications"] / 81 +
        data["number_diagnoses"] / 16
    )
    data["historique_soins"] = data["number_inpatient"] + data["number_emergency"]
    data["long_sejour"] = (data["time_in_hospital"] > 4).astype(int)
    data["patient_frequent"] = (data["number_outpatient"] > 0).astype(int)
    for col in feature_names:
        if col not in data.columns:
            data[col] = 0
    return data[feature_names]

def get_risk(proba):
    if proba >= 0.5:   return "CRITIQUE", "#dc2626", "critique"
    if proba >= 0.3:   return "ELEVE",    "#f97316", "eleve"
    if proba >= 0.15:  return "MODERE",   "#eab308", "modere"
    return "FAIBLE", "#22c55e", "faible"

# ── HEADER ───────────────────────────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("## 🏥")
with col_title:
    st.title("MedRisk — Prediction de Readmission Hospitaliere")
    st.caption(f"Modele XGBoost · AUC-ROC : {config['auc_roc']} · Seuil : {THRESHOLD:.2f} · Version {config['model_version']}")

st.divider()

# ── TABS PRINCIPAUX ───────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Analyse d'un patient", "Analyse en lot", "Informations du modele"])

# ══════════════════════════════════════════════════════════
# TAB 1 — ANALYSE D'UN PATIENT
# ══════════════════════════════════════════════════════════
with tab1:
    # Formulaire patient dans la sidebar
    with st.sidebar:
        st.header("Profil du patient")
        st.caption("Renseignez les informations cliniques")

        with st.expander("Informations generales", expanded=True):
            age = st.selectbox("Tranche d'age", [
                "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
            ], index=6)
            gender = st.selectbox("Genre", ["Male", "Female"])

        with st.expander("Sejour hospitalier", expanded=True):
            time_in_hospital = st.slider("Duree du sejour (jours)", 1, 14, 5)
            num_lab_procedures = st.slider("Examens biologiques", 1, 132, 45)
            num_procedures = st.slider("Actes medicaux", 0, 6, 2)
            num_medications = st.slider("Medicaments prescrits", 1, 81, 15)
            number_diagnoses = st.slider("Nombre de diagnostics", 1, 16, 7)

        with st.expander("Historique medical", expanded=True):
            number_outpatient = st.number_input("Visites ambulatoires anterieures", 0, 20, 0)
            number_emergency = st.number_input("Passages urgences anterieurs", 0, 20, 0)
            number_inpatient = st.number_input("Hospitalisations anterieures", 0, 20, 0)

        with st.expander("Bilan diabete", expanded=True):
            max_glu_serum = st.selectbox("Glycemie serique", ["None", "Norm", ">200", ">300"])
            A1Cresult = st.selectbox("Resultat HbA1c", ["None", "Norm", ">7", ">8"])
            insulin = st.selectbox("Traitement insuline", ["No", "Steady", "Up", "Down"])
            diabetesMed = st.selectbox("Medicament diabete prescrit", ["Yes", "No"])

        predict_btn = st.button("Analyser ce patient", use_container_width=True, type="primary")

    # Resultats
    if predict_btn:
        patient = {
            "age": age, "gender": gender,
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": int(number_outpatient),
            "number_emergency": int(number_emergency),
            "number_inpatient": int(number_inpatient),
            "number_diagnoses": number_diagnoses,
            "max_glu_serum": max_glu_serum,
            "A1Cresult": A1Cresult,
            "insulin": insulin,
            "diabetesMed": diabetesMed
        }

        with st.spinner("Calcul en cours..."):
            X = preprocess(patient)
            proba = float(model.predict_proba(X)[0, 1])
            readmitted = proba >= THRESHOLD
            risk_label, risk_color, risk_class = get_risk(proba)

            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X)
            sv_arr = sv[1] if isinstance(sv, list) else sv

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Impact": sv_arr[0],
                "Valeur": X.values[0]
            }).sort_values("Impact", key=abs, ascending=False)

        # Ligne de KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Probabilite readmission", f"{proba:.1%}")
        with k2:
            st.metric("Niveau de risque", risk_label)
        with k3:
            st.metric("Decision", "READMISSION PROBABLE" if readmitted else "RISQUE CONTROLE")
        with k4:
            st.metric("Seuil utilise", f"{THRESHOLD:.2f}")

        st.divider()

        # Graphiques
        col_gauge, col_shap = st.columns([1, 1])

        with col_gauge:
            st.subheader("Score de risque")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=proba * 100,
                title={"text": "Probabilite de readmission (%)"},
                delta={"reference": THRESHOLD * 100, "suffix": "% (seuil)"},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar": {"color": risk_color, "thickness": 0.3},
                    "steps": [
                        {"range": [0, 15],  "color": "#dcfce7"},
                        {"range": [15, 30], "color": "#fef9c3"},
                        {"range": [30, 50], "color": "#ffedd5"},
                        {"range": [50, 100],"color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#1e293b", "width": 3},
                        "thickness": 0.8,
                        "value": THRESHOLD * 100
                    }
                },
                number={"suffix": "%", "font": {"size": 36}}
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Recommandation
            if readmitted:
                st.markdown(f"""<div class="risk-{risk_class}">
                <b>Recommendation clinique</b><br/>
                Surveillance renforcee recommandee<br/>
                Suivi prevu : J+7 et J+15 apres la sortie<br/>
                Coordination avec le medecin traitant avant la sortie
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="risk-{risk_class}">
                <b>Recommendation clinique</b><br/>
                Risque controle — suivi standard suffisant<br/>
                Consultation de controle a J+30<br/>
                Renouvellement du traitement prescrit
                </div>""", unsafe_allow_html=True)

        with col_shap:
            st.subheader("Facteurs cliniques determinants")

            top8 = shap_df.head(8)
            colors_bar = ["#dc2626" if v > 0 else "#22c55e" for v in top8["Impact"]]

            fig_shap = go.Figure(go.Bar(
                x=top8["Impact"],
                y=top8["Feature"],
                orientation="h",
                marker_color=colors_bar,
                text=[f"{v:+.3f}" for v in top8["Impact"]],
                textposition="outside"
            ))
            fig_shap.update_layout(
                height=280,
                xaxis_title="Impact sur le score (SHAP)",
                margin=dict(t=10, b=10, l=10, r=60),
                xaxis=dict(zeroline=True, zerolinecolor="#94a3b8", zerolinewidth=2)
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("Rouge = augmente le risque · Vert = diminue le risque")

        # Detail des facteurs
        st.subheader("Detail des 5 facteurs les plus importants")
        top5 = shap_df.head(5).copy()
        top5["Direction"] = top5["Impact"].apply(lambda x: "Augmente le risque" if x > 0 else "Diminue le risque")
        top5["Impact absolu"] = top5["Impact"].abs().round(4)
        top5["Valeur"] = top5["Valeur"].round(3)
        top5 = top5[["Feature", "Valeur", "Impact", "Impact absolu", "Direction"]]
        st.dataframe(top5.reset_index(drop=True), use_container_width=True)

    else:
        st.info("Renseignez le profil du patient dans le panneau de gauche et cliquez sur 'Analyser ce patient'")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Comment utiliser ce dashboard")
            st.markdown("""
1. Remplissez les informations cliniques dans le panneau gauche
2. Cliquez sur **Analyser ce patient**
3. Consultez le score de risque et les facteurs determinants
4. Suivez la recommandation clinique affichee
            """)
        with col_b:
            st.markdown("#### Niveaux de risque")
            st.markdown("""
| Niveau | Probabilite | Action |
|--------|-------------|--------|
| FAIBLE | < 15% | Suivi standard J+30 |
| MODERE | 15-30% | Consultation J+14 |
| ELEVE  | 30-50% | Suivi J+7 |
| CRITIQUE | > 50% | Suivi J+3 et J+7 |
            """)

# ══════════════════════════════════════════════════════════
# TAB 2 — ANALYSE EN LOT
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Analyse en lot — Importer un fichier CSV")
    st.caption("Le fichier doit contenir les memes colonnes que le dataset d'entrainement")

    uploaded_file = st.file_uploader("Importer un fichier CSV de patients", type=["csv"])

    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        st.write(f"Fichier charge : {len(df_upload)} patients")
        st.dataframe(df_upload.head(5), use_container_width=True)

        if st.button("Analyser tous les patients", type="primary"):
            with st.spinner(f"Analyse de {len(df_upload)} patients..."):
                results = []
                for _, row in df_upload.iterrows():
                    try:
                        data_r = row.to_dict()
                        X_r = preprocess(data_r)
                        proba_r = float(model.predict_proba(X_r)[0, 1])
                        risk_l, risk_c, _ = get_risk(proba_r)
                        results.append({
                            "probabilite": round(proba_r, 4),
                            "readmission_predite": int(proba_r >= THRESHOLD),
                            "niveau_risque": risk_l
                        })
                    except Exception:
                        results.append({
                            "probabilite": None,
                            "readmission_predite": None,
                            "niveau_risque": "ERREUR"
                        })

            df_results = pd.concat([df_upload.reset_index(drop=True),
                                     pd.DataFrame(results)], axis=1)

            st.success(f"Analyse terminee — {sum(r['readmission_predite'] or 0 for r in results)} patients a risque identifies")

            # KPIs globaux
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total patients", len(results))
            c2.metric("Readmissions predites", sum(r['readmission_predite'] or 0 for r in results))
            c3.metric("Taux de readmission predit", f"{sum(r['readmission_predite'] or 0 for r in results)/len(results):.1%}")
            c4.metric("Probabilite moyenne", f"{np.mean([r['probabilite'] for r in results if r['probabilite']]):.1%}")

            # Distribution des niveaux de risque
            risk_counts = pd.DataFrame(results)["niveau_risque"].value_counts()
            fig_risk = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Repartition par niveau de risque",
                color=risk_counts.index,
                color_discrete_map={"FAIBLE": "#22c55e", "MODERE": "#eab308",
                                    "ELEVE": "#f97316", "CRITIQUE": "#dc2626"}
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            # Tableau des resultats
            st.subheader("Resultats detailles")
            st.dataframe(df_results, use_container_width=True)

            # Export
            csv_export = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Telecharger les resultats (CSV)",
                csv_export,
                "medrisk_resultats.csv",
                "text/csv"
            )
    else:
        st.info("Importez un fichier CSV pour analyser plusieurs patients en une seule fois")
        st.markdown("**Format attendu :** memes colonnes que `diabetes_readmission.csv`")

        # Demo avec le dataset d'entrainement
        st.subheader("Demo — Simuler avec des patients aleatoires")
        n_demo = st.slider("Nombre de patients a generer", 10, 200, 50)
        if st.button("Generer et analyser"):
            np.random.seed(42)
            demo_data = {
                'age': np.random.choice(['[50-60)', '[60-70)', '[70-80)', '[80-90)'], n_demo),
                'gender': np.random.choice(['Male', 'Female'], n_demo),
                'time_in_hospital': np.random.randint(1, 14, n_demo),
                'num_lab_procedures': np.random.randint(10, 120, n_demo),
                'num_procedures': np.random.randint(0, 6, n_demo),
                'num_medications': np.random.randint(5, 70, n_demo),
                'number_outpatient': np.random.poisson(0.4, n_demo),
                'number_emergency': np.random.poisson(0.3, n_demo),
                'number_inpatient': np.random.poisson(0.6, n_demo),
                'number_diagnoses': np.random.randint(3, 14, n_demo),
                'max_glu_serum': np.random.choice(['None', 'Norm', '>200', '>300'], n_demo),
                'A1Cresult': np.random.choice(['None', 'Norm', '>7', '>8'], n_demo),
                'insulin': np.random.choice(['No', 'Steady', 'Up', 'Down'], n_demo),
                'diabetesMed': np.random.choice(['Yes', 'No'], n_demo, p=[0.77, 0.23])
            }
            df_demo = pd.DataFrame(demo_data)

            probas = []
            for _, row in df_demo.iterrows():
                try:
                    X_d = preprocess(row.to_dict())
                    probas.append(float(model.predict_proba(X_d)[0, 1]))
                except:
                    probas.append(0.0)

            df_demo['probabilite_readmission'] = probas
            df_demo['niveau_risque'] = [get_risk(p)[0] for p in probas]
            df_demo['readmission_predite'] = [int(p >= THRESHOLD) for p in probas]

            c1, c2, c3 = st.columns(3)
            c1.metric("Patients analyses", n_demo)
            c2.metric("Readmissions predites", df_demo['readmission_predite'].sum())
            c3.metric("Taux predit", f"{df_demo['readmission_predite'].mean():.1%}")

            # Distribution des probabilites
            fig_hist = px.histogram(df_demo, x='probabilite_readmission',
                                     color='niveau_risque', nbins=30,
                                     title="Distribution des probabilites de readmission",
                                     color_discrete_map={"FAIBLE": "#22c55e", "MODERE": "#eab308",
                                                         "ELEVE": "#f97316", "CRITIQUE": "#dc2626"})
            st.plotly_chart(fig_hist, use_container_width=True)
            st.dataframe(df_demo.sort_values('probabilite_readmission', ascending=False).head(20),
                          use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — INFORMATIONS DU MODELE
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("Informations sur le modele MedRisk")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithme", "XGBoost")
    c2.metric("AUC-ROC", config['auc_roc'])
    c3.metric("Seuil de decision", f"{THRESHOLD:.2f}")
    c4.metric("Version", config['model_version'])

    st.divider()

    col_info, col_feat = st.columns([1, 1])

    with col_info:
        st.markdown("#### Description du modele")
        st.markdown(f"""
- **Type :** Classification binaire (readmission < 30 jours)
- **Algorithme :** XGBoost avec gestion du desequilibre de classes
- **Seuil :** {THRESHOLD:.2f} (optimise pour maximiser le rappel)
- **AUC-ROC :** {config['auc_roc']} (capacite a distinguer les cas)
- **Rappel cible :** >= 70% sur la classe minoritaire
- **Dataset :** Diabetes 130-US Hospitals (inspire UCI)

#### Niveaux de risque
| Niveau | Seuil | Interpretation |
|--------|-------|----------------|
| FAIBLE | < 15% | Pas d'action particuliere |
| MODERE | 15-30% | Vigilance recommandee |
| ELEVE  | 30-50% | Suivi renforce |
| CRITIQUE | > 50% | Intervention immediate |

#### Avertissement medical
Ce modele est un **outil d'aide a la decision** uniquement.
Il ne remplace pas le jugement clinique du medecin.
Toute decision therapeutique releve de la responsabilite exclusive du praticien.
        """)

    with col_feat:
        st.markdown("#### Features utilisees par le modele")
        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Index": range(len(feature_names))
        })
        st.dataframe(feat_df, use_container_width=True, height=400)

        st.markdown("#### Features creees (Feature Engineering)")
        st.markdown("""
| Feature | Calcul | Interpretation |
|---------|--------|----------------|
| score_complexite | lab/132 + med/81 + diag/16 | Indice de complexite clinique |
| historique_soins | inpatient + emergency | Frequence d'utilisation des soins |
| long_sejour | time > 4 jours | Sejour superieur a la mediane |
| patient_frequent | outpatient > 0 | Patient avec suivi ambulatoire |
        """)

