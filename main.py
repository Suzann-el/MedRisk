"""
MedRisk API — Prédiction de réadmission hospitalière < 30 jours
Modèle : XGBoost | Dataset : UCI Diabetes 130-US Hospitals

Lancer : uvicorn main:app --reload
Docs   : http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import shap

# ── Chargement du modèle au démarrage ────────────────────────────────────────
# Les fichiers sont chargés une seule fois en mémoire au lancement de l'API
app = FastAPI(
    title="MedRisk API",
    description="Prédiction du risque de réadmission hospitalière < 30 jours — XGBoost",
    version="1.0.0"
)

try:
    model         = joblib.load("medrisk_model.pkl")
    feature_names = joblib.load("medrisk_features.pkl")
    with open("medrisk_config.json") as f:
        config = json.load(f)
    THRESHOLD = config["threshold"]
    explainer = shap.TreeExplainer(model)
    print(f"✅ Modèle chargé — AUC-ROC : {config['auc_roc']} | Seuil : {THRESHOLD}")
except Exception as e:
    raise RuntimeError(f"Impossible de charger le modèle : {e}")


# ── Schéma du patient (validation automatique par Pydantic) ──────────────────
class PatientProfile(BaseModel):
    age:                str   = Field(..., example="[60-70)")
    gender:             str   = Field(..., example="Male")
    time_in_hospital:   int   = Field(..., ge=1, le=14, example=5)
    num_lab_procedures: int   = Field(..., ge=0, example=45)
    num_procedures:     int   = Field(..., ge=0, example=2)
    num_medications:    int   = Field(..., ge=0, example=15)
    number_outpatient:  int   = Field(0,   ge=0, example=0)
    number_emergency:   int   = Field(0,   ge=0, example=1)
    number_inpatient:   int   = Field(0,   ge=0, example=2)
    number_diagnoses:   int   = Field(..., ge=1, example=7)
    max_glu_serum:      str   = Field("None", example="None")
    A1Cresult:          str   = Field("None", example=">8")
    insulin:            str   = Field("No",   example="Steady")
    diabetesMed:        str   = Field("Yes",  example="Yes")


# ── Preprocessing identique à celui du notebook ───────────────────────────────
def preprocess(patient: PatientProfile) -> pd.DataFrame:
    """
    Applique exactement le même preprocessing que dans le notebook :
    1. Encodage ordinal de l'âge
    2. LabelEncoder sur les colonnes object
    3. Feature engineering (4 features dérivées)
    """
    data = pd.DataFrame([patient.dict()])

    # 1. Encodage ordinal de l'âge
    age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                 '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    age_map = {v: i for i, v in enumerate(age_order)}
    data['age'] = data['age'].map(age_map)

    # 2. LabelEncoder sur les colonnes object restantes
    le = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = le.fit_transform(data[col].astype(str))

    # 3. Feature engineering — même formules que dans le notebook
    data['score_complexite'] = (
        data['num_lab_procedures'] / 132 +
        data['num_medications']    / 81  +
        data['number_diagnoses']   / 16
    )
    data['historique_soins'] = (
        data['number_inpatient'] + data['number_emergency']
    )
    data['long_sejour']      = (data['time_in_hospital'] > 4).astype(int)
    data['patient_frequent'] = (data['number_outpatient'] > 0).astype(int)

    # 4. Aligner les colonnes avec celles du modèle entraîné
    for col in feature_names:
        if col not in data.columns:
            data[col] = 0

    return data[feature_names]


def get_risk_level(proba: float) -> str:
    """Convertit une probabilité en niveau de risque clinique."""
    if proba >= 0.50: return "CRITIQUE"
    if proba >= 0.30: return "ÉLEVÉ"
    if proba >= 0.10: return "MODÉRÉ"
    return "FAIBLE"


def get_recommendation(readmitted: bool, risk_level: str) -> str:
    """Retourne la recommandation clinique selon le niveau de risque."""
    reco = {
        "CRITIQUE": "Intervention prioritaire — coordination médecin avant la sortie, contact à J+3",
        "ÉLEVÉ":    "Surveillance renforcée — contact téléphonique à J+7 et J+15",
        "MODÉRÉ":   "Vigilance — consultation recommandée à J+14",
        "FAIBLE":   "Suivi standard — consultation à J+30 chez le médecin traitant",
    }
    return reco.get(risk_level, "Suivi standard")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Informations générales sur le service."""
    return {
        "service":         "MedRisk API",
        "modele":          "XGBoost",
        "version":         config["model_version"],
        "auc_roc":         config["auc_roc"],
        "seuil_decision":  THRESHOLD,
        "documentation":   "/docs"
    }


@app.get("/health")
def health():
    """Vérification que l'API et le modèle sont opérationnels."""
    return {
        "status":        "ok",
        "model_loaded":  model is not None,
        "n_features":    len(feature_names),
    }


@app.post("/predict")
def predict(patient: PatientProfile):
    """
    Prédit le risque de réadmission pour un patient.

    Retourne :
    - probabilite : score de risque entre 0 et 1
    - readmission_predite : 1 si réadmission probable, 0 sinon
    - niveau_risque : FAIBLE / MODÉRÉ / ÉLEVÉ / CRITIQUE
    - top_3_facteurs : les 3 features les plus influentes (SHAP)
    - recommandation : action clinique recommandée
    """
    try:
        # Preprocessing
        X = preprocess(patient)

        # Prédiction
        proba      = float(model.predict_proba(X)[0, 1])
        readmitted = int(proba >= THRESHOLD)
        risk_level = get_risk_level(proba)

        # Valeurs SHAP — top 3 facteurs influents
        sv = explainer.shap_values(X)

        # XGBoost peut retourner un array 2D ou 3D selon la version
        if isinstance(sv, list):
            sv_arr = sv[1]
        elif hasattr(sv, 'ndim') and sv.ndim == 3:
            sv_arr = sv[:, :, 1]
        else:
            sv_arr = sv

        shap_df = pd.DataFrame({
            "feature":  feature_names,
            "impact":   sv_arr[0],
            "valeur":   X.values[0]
        }).sort_values("impact", key=abs, ascending=False)

        top_3 = [
            {
                "feature":   row["feature"],
                "impact":    round(float(row["impact"]), 4),
                "direction": "augmente le risque" if row["impact"] > 0 else "diminue le risque",
                "valeur":    round(float(row["valeur"]), 2)
            }
            for _, row in shap_df.head(3).iterrows()
        ]

        return {
            "probabilite":        round(proba, 4),
            "readmission_predite": readmitted,
            "niveau_risque":       risk_level,
            "seuil_utilise":       THRESHOLD,
            "top_3_facteurs":      top_3,
            "recommandation":      get_recommendation(bool(readmitted), risk_level)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(patients: list[PatientProfile]):
    """
    Prédit le risque de réadmission pour une liste de patients.
    Utile pour analyser un groupe de patients à la sortie de l'hôpital.
    """
    if len(patients) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 1000 patients par requête batch."
        )

    results = []
    for i, p in enumerate(patients):
        try:
            result = predict(p)
            result["patient_index"] = i
            results.append(result)
        except Exception as e:
            results.append({
                "patient_index": i,
                "erreur": str(e)
            })

    n_readmis = sum(r.get("readmission_predite", 0) for r in results)

    return {
        "total_patients":    len(results),
        "readmissions_predites": n_readmis,
        "taux_predit":       round(n_readmis / len(results) * 100, 1),
        "predictions":       results
    }
