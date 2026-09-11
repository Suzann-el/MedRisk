# 🏥 MedRisk — Prédiction de Réadmission Hospitalière sous 30 jours

## Description
Modèle de machine learning pour identifier les patients diabétiques à risque d'être
réadmis à l'hôpital dans les 30 jours suivant leur sortie.
Ce projet prolonge directement l'expérience au CHRU de Tours et démontre
une maîtrise des données cliniques complexes.

## Dataset
**Diabetes 130-US Hospitals** — données cliniques synthétiques
inspirées du dataset UCI (Clore et al., 1994-1999)
- Fichier : `diabetes_readmission.csv`
- 5000 patients, 15 variables cliniques
- Taux de réadmission : ~6-8% (classe très minoritaire → défi ML)

### Variables
| Variable | Description |
|----------|-------------|
| age | Tranche d'âge [0-10), [10-20)... [90-100) |
| gender | Genre du patient |
| time_in_hospital | Durée du séjour (jours) |
| num_lab_procedures | Nombre de tests biologiques |
| num_procedures | Nombre d'actes médicaux |
| num_medications | Nombre de médicaments |
| number_outpatient | Visites ambulatoires avant |
| number_emergency | Passages aux urgences avant |
| number_inpatient | Hospitalisations antérieures |
| number_diagnoses | Nombre de diagnostics |
| max_glu_serum | Résultat test glycémie |
| A1Cresult | Résultat HbA1c |
| insulin | Traitement insuline |
| diabetesMed | Médicament diabète prescrit |
| **readmitted** | **Cible : réadmis <30j (1=oui, 0=non)** |

## Stack technique
```
Python · Pandas · NumPy · Scikit-learn
XGBoost · SMOTE (imbalanced-learn)
SHAP · Matplotlib · Seaborn · Plotly
```

## Installation
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn plotly
```

## Structure du notebook
| Section | Contenu |
|---------|---------|
| 0. Imports | Configuration |
| 1. Chargement | Exploration initiale, taux de réadmission |
| 2. Preprocessing | Encodage des variables cliniques |
| 3. Feature Engineering | Score de complexité, historique de soins |
| 4. Modélisation | SMOTE + XGBoost + scale_pos_weight |
| 5. SHAP | Interprétabilité pour équipes médicales |
| 6. Sous-groupes | Analyse par niveau de risque |
| 7. Prochaines étapes | Rapport PDF, dashboard médical |

## Résultats attendus
- AUC-ROC : ~0.72-0.78
- Rappel (classe réadmis) : ~0.65
- Note : classe très déséquilibrée → accuracy trompeuse, se fier à l'AUC et au rappel

## Points clés à comprendre
1. **Déséquilibre extrême** (~6%) → SMOTE + scale_pos_weight indispensables
2. **Rappel > Précision** — en médecine, manquer un cas (FN) est plus grave que sur-alerter (FP)
3. **SHAP pour les médecins** — traduire les prédictions en facteurs cliniques compréhensibles
4. **Validation externe** — toujours tester sur un hôpital différent avant de déployer

## Pour aller plus loin (sur données réelles)
Accès gratuit aux données MIMIC après inscription :
https://physionet.org/content/mimiciii/1.4/

```python
# Variables supplémentaires disponibles dans MIMIC :
# - Diagnostics CIM-10 complets
# - Résultats biologiques détaillés
# - Notes médicales (NLP possible)
# - Prescriptions médicamenteuses
```
