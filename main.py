import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import logging

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


import optuna
from optuna.pruners import MedianPruner

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import shap
import matplotlib.pyplot as plt

# ========================================================================
# CONFIG GLOBALE
# ========================================================================
warnings.filterwarnings('ignore')
logging.getLogger('lightgbm').setLevel(logging.CRITICAL)

RSEED = 42
np.random.seed(RSEED)

data_path = Path("data")

# ========================================================================
# ETAPE 1 : LECTURE DES DONNEES
# ========================================================================
train = pd.read_csv(data_path / "train.csv")
test = pd.read_csv(data_path / "test.csv")

print(f"Taille du training set : {len(train)} lignes")
print(f"Taille du test set     : {len(test)} lignes")

X_orig = train.drop(columns=["y"])
y = train["y"].copy()
test_id = test["id"].copy()
X_test_orig = test.drop(columns=["id"])



# Colonnes catégorielles
cat_cols = ["country", "opsys", "acquis"]

# ========================================================================
# ETAPE 2 : FEATURE ENGINEERING V2
# ========================================================================
def create_features_v2(df):
    df = df.copy()

    # --- Features simples de densité / ratios ---
    df["score_per_session"] = df["totscore"] / (df["numsessions"] + 1)
    df["playtime_per_session"] = df["totplaytime"] / (df["numsessions"] + 1)
    df["purchases_per_session"] = df["totpurchases"] / (df["numsessions"] + 1)

    df["score_per_playtime"] = df["totscore"] / (df["totplaytime"] + 1)
    df["lives_per_session"] = df["numlives"] / (df["numsessions"] + 1)
    df["elements_per_session"] = df["numelements"] / (df["numsessions"] + 1)

    df["playtime_per_day"] = df["totplaytime"] / (df["numdays"] + 1)
    df["sessions_per_day"] = df["numsessions"] / (df["numdays"] + 1)

    df["purchases_per_day"] = df["totpurchases"] / (df["numdays"] + 1)
    df["avg_purchase_value"] = df["totpurchases"] / (df["numpurchases"] + 1)
    df["purchase_rate"] = df["numpurchases"] / (df["numsessions"] + 1)

    # --- Interactions de compétences ---
    df["skill_product"] = df["skill1"] * df["skill2"]
    df["skill_ratio"] = df["skill1"] / (df["skill2"] + 0.01)
    df["skill_sum"] = df["skill1"] + df["skill2"]
    df["skill_diff"] = df["skill1"] - df["skill2"]

    # --- Ajustement par difficulté ---
    df["score_per_difficulty"] = df["totscore"] / (df["difflevel"] + 1)
    df["lives_per_difficulty"] = df["numlives"] / (df["difflevel"] + 1)
    df["elements_per_difficulty"] = df["numelements"] / (df["difflevel"] + 1)

    df["score_per_life"] = df["totscore"] / (df["numlives"] + 1)
    df["elements_per_life"] = df["numelements"] / (df["numlives"] + 1)

    # --- Interactions autour de totpurchases / trendpurchase / numpurchases ---
    df["purchases_x_trendpurchase"] = df["totpurchases"] * df["trendpurchase"]
    df["purchases_x_numpurchases"] = df["totpurchases"] * df["numpurchases"]
    df["purchases_x_trendsession"] = df["totpurchases"] * df["trendsession"]
    df["purchases_x_score"] = df["totpurchases"] * df["totscore"]
    df["purchases_x_sessions"] = df["totpurchases"] * df["numsessions"]

    df["trendpurchase_x_trendsession"] = df["trendpurchase"] * df["trendsession"]
    df["trendpurchase_x_numpurchases"] = df["trendpurchase"] * df["numpurchases"]
    df["trendpurchase_x_sessions"] = df["trendpurchase"] * df["numsessions"]

    df["numpurchases_x_trendsession"] = df["numpurchases"] * df["trendsession"]
    df["numpurchases_x_sessions"] = df["numpurchases"] * df["numsessions"]

    df["purchases_per_score"] = df["totpurchases"] / (df["totscore"] + 1)
    df["purchases_per_playtime"] = df["totpurchases"] / (df["totplaytime"] + 1)
    df["purchases_per_lives"] = df["totpurchases"] / (df["numlives"] + 1)

    df["skill1_x_purchases"] = df["skill1"] * df["totpurchases"]
    df["skill2_x_purchases"] = df["skill2"] * df["totpurchases"]
    df["skill_total_x_purchases"] = (df["skill1"] + df["skill2"]) * df["totpurchases"]

    df["skill1_x_score"] = df["skill1"] * df["totscore"]
    df["skill2_x_score"] = df["skill2"] * df["totscore"]
    df["skill1_x_playtime"] = df["skill1"] * df["totplaytime"]

    # --- Puissances / racines ---
    df["purchases_squared"] = df["totpurchases"] ** 2
    df["trendpurchase_squared"] = df["trendpurchase"] ** 2
    df["numpurchases_squared"] = df["numpurchases"] ** 2
    df["trendsession_squared"] = df["trendsession"] ** 2
    df["score_squared"] = df["totscore"] ** 2

    df["purchases_sqrt"] = np.sqrt(df["totpurchases"] + 1)
    df["score_sqrt"] = np.sqrt(df["totscore"] + 1)
    df["playtime_sqrt"] = np.sqrt(df["totplaytime"] + 1)

    # --- Binnings sur totpurchases / score ---
    purchases_q25 = df["totpurchases"].quantile(0.25)
    purchases_q50 = df["totpurchases"].quantile(0.50)
    purchases_q75 = df["totpurchases"].quantile(0.75)

    df["purchases_very_low"] = (df["totpurchases"] <= purchases_q25).astype(int)
    df["purchases_low"] = ((df["totpurchases"] > purchases_q25) &
                           (df["totpurchases"] <= purchases_q50)).astype(int)
    df["purchases_high"] = ((df["totpurchases"] > purchases_q50) &
                            (df["totpurchases"] <= purchases_q75)).astype(int)
    df["purchases_very_high"] = (df["totpurchases"] > purchases_q75).astype(int)

    score_q75 = df["totscore"].quantile(0.75)
    df["score_high"] = (df["totscore"] > score_q75).astype(int)

    # --- Scores d'engagement globaux ---
    df["engagement_score"] = (df["totplaytime"] * df["totscore"] * df["numsessions"]) / (df["numdays"] + 1)
    df["monetization_intensity"] = df["totpurchases"] / (df["totplaytime"] + 1)
    df["activity_intensity"] = (df["numsessions"] * df["totscore"]) / (df["numdays"] + 1)

    # --- Trends combinés ---
    df["trend_ratio"] = df["trendsession"] / (df["trendpurchase"] + 0.01)
    df["trend_product"] = df["trendsession"] * df["trendpurchase"]
    df["trend_sum"] = df["trendsession"] + df["trendpurchase"]

    # --- Efficacité globale ---
    df["efficiency_score"] = (df["totscore"] * df["numelements"]) / (df["totplaytime"] * df["numlives"] + 1)
    df["purchase_efficiency"] = df["totpurchases"] / (df["numsessions"] * df["numdays"] + 1)

    # Remplacement des infinis / NaN

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

print("\nCréation des features V2...")
X_fe = create_features_v2(X_orig)
X_test_fe = create_features_v2(X_test_orig)

print(f"Nombre de features originales : {X_orig.shape[1]}")
print(f"Nombre de features V2         : {X_fe.shape[1]}")
print(f"Nouvelles features ajoutées   : {X_fe.shape[1] - X_orig.shape[1]}")

# ========================================================================
# ETAPE 3 : ENCODAGE + SPLIT TRAIN / VALIDATION
# ========================================================================
le = LabelEncoder()
y_enc = le.fit_transform(y)

idx_train, idx_val = train_test_split(
    np.arange(len(X_fe)),
    test_size=0.2,
    random_state=RSEED,
    stratify=y_enc
)

X_train_fe = X_fe.iloc[idx_train].reset_index(drop=True)
X_val_fe = X_fe.iloc[idx_val].reset_index(drop=True)
y_train = y_enc[idx_train]
y_val = y_enc[idx_val]

# Pour XGBoost / CatBoost : on utilise X_fe (numérique)
X_train = X_train_fe.copy()
X_val = X_val_fe.copy()

# Pour LightGBM : on convertit les colonnes catégorielles en "category"
X_train_lgb = X_train_fe.copy()
X_val_lgb = X_val_fe.copy()
X_fe_lgb = X_fe.copy()
X_test_fe_lgb = X_test_fe.copy()

for c in cat_cols:
    if c in X_train_lgb.columns:
        X_train_lgb[c] = X_train_lgb[c].astype("category")
        X_val_lgb[c] = X_val_lgb[c].astype("category")
        X_fe_lgb[c] = X_fe_lgb[c].astype("category")
        X_test_fe_lgb[c] = X_test_fe_lgb[c].astype("category")

# Pour CatBoost : indices des colonnes catégorielles
cat_features_idx = [X_train.columns.get_loc(c) for c in cat_cols]

print(f"\nTaille train : {len(X_train)}, validation : {len(X_val)}")
print("Données préparées pour LightGBM, XGBoost et CatBoost.")


# ========================================================================
# ETAPE 4 : OPTIMISATION LIGHTGBM AVEC OPTUNA
# ========================================================================
print("\n" + "="*70)
print("LIGHTGBM - Optimisation avec Optuna")
print("="*70)

def objective_lgbm(trial):
    param = {
        "objective": "multiclass",
        "num_class": 4,
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.03, 0.05, 0.07, 0.1]),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "num_leaves": trial.suggest_int("num_leaves", 31, 80),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 60),
        "subsample": trial.suggest_categorical("subsample", [0.7, 0.8, 0.9]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.7, 0.8, 0.9, 1.0]),
        "reg_lambda": trial.suggest_categorical("reg_lambda", [0.5, 1.0, 2.0, 5.0]),
        "reg_alpha": trial.suggest_categorical("reg_alpha", [0.0, 0.1, 0.5, 1.0]),
        "subsample_freq": 1,
        "random_state": RSEED,
        "n_jobs": -1,
        "verbosity": -1
    }

    model = LGBMClassifier(**param)

    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RSEED)
        scores = cross_val_score(
            model, X_train_lgb, y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            error_score="raise"
        )
        return scores.mean()
    except Exception as e:
        print(f"Trial {trial.number} échoue : {e}")
        return 0.0

optuna.logging.set_verbosity(optuna.logging.WARNING)
study_lgbm = optuna.create_study(direction="maximize", pruner=MedianPruner())
study_lgbm.optimize(objective_lgbm, n_trials=60, show_progress_bar=True)

print("\n" + "="*70)
print("LIGHTGBM - Meilleurs hyperparamètres")
print("="*70)
for k, v in study_lgbm.best_params.items():
    print(f"  {k}: {v}")
print(f"\nMeilleur score CV 5-fold : {study_lgbm.best_value:.4f}")

best_params_lgbm = study_lgbm.best_params.copy()

lgb_final = LGBMClassifier(
    objective="multiclass",
    num_class=4,
    subsample_freq=1,
    random_state=RSEED,
    n_jobs=-1,
    verbosity=-1,
    **best_params_lgbm
)

print("\nEntraînement du modèle final LightGBM...")
lgb_final.fit(
    X_train_lgb, y_train,
    eval_set=[(X_val_lgb, y_val)],
    callbacks=[early_stopping(stopping_rounds=50),
               log_evaluation(period=0)],
    categorical_feature=cat_cols
)

y_val_proba_lgb = lgb_final.predict_proba(X_val_lgb)
y_val_pred_lgb = np.argmax(y_val_proba_lgb, axis=1)

acc_lgb = accuracy_score(y_val, y_val_pred_lgb)
f1m_lgb = f1_score(y_val, y_val_pred_lgb, average="macro")
auc_lgb = roc_auc_score(y_val, y_val_proba_lgb, multi_class="ovr")

print("\n" + "="*70)
print("LIGHTGBM - Performance validation")
print("="*70)
print(f"Accuracy  : {acc_lgb:.4f}")
print(f"F1-macro : {f1m_lgb:.4f}")
print(f"ROC-AUC  : {auc_lgb:.4f}")


# ========================================================================
# ETAPE 5 : OPTIMISATION XGBOOST AVEC OPTUNA
# ========================================================================
print("\n" + "="*70)
print("XGBOOST - Optimisation avec Optuna")
print("="*70)

def objective_xgb(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.03, 0.05, 0.07, 0.1]),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_categorical("subsample", [0.7, 0.8, 0.9]),
        "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.7, 0.8, 0.9, 1.0]),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
        "reg_lambda": trial.suggest_categorical("reg_lambda", [0.5, 1.0, 2.0, 5.0]),
        "reg_alpha": trial.suggest_categorical("reg_alpha", [0.0, 0.1, 0.5, 1.0]),
        "objective": "multi:softprob",
        "num_class": 4,
        "random_state": RSEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0
    }

    model = XGBClassifier(**param)

    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RSEED)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            error_score="raise"
        )
        return scores.mean()
    except Exception as e:
        print(f"Trial {trial.number} échoue : {e}")
        return 0.0

study_xgb = optuna.create_study(direction="maximize", pruner=MedianPruner())
study_xgb.optimize(objective_xgb, n_trials=60, show_progress_bar=True)

print("\n" + "="*70)
print("XGBOOST - Meilleurs hyperparamètres")
print("="*70)
for k, v in study_xgb.best_params.items():
    print(f"  {k}: {v}")
print(f"\nMeilleur score CV 5-fold : {study_xgb.best_value:.4f}")

best_params_xgb = study_xgb.best_params.copy()

xgb_final = XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    random_state=RSEED,
    n_jobs=-1,
    tree_method="hist",
    verbosity=0,
    early_stopping_rounds=50,
    **best_params_xgb
)

print("\nEntraînement du modèle final XGBoost...")
xgb_final.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

y_val_proba_xgb = xgb_final.predict_proba(X_val)
y_val_pred_xgb = np.argmax(y_val_proba_xgb, axis=1)

acc_xgb = accuracy_score(y_val, y_val_pred_xgb)
f1m_xgb = f1_score(y_val, y_val_pred_xgb, average="macro")
auc_xgb = roc_auc_score(y_val, y_val_proba_xgb, multi_class="ovr")

print("\n" + "="*70)
print("XGBOOST - Performance validation")
print("="*70)
print(f"Accuracy  : {acc_xgb:.4f}")
print(f"F1-macro : {f1m_xgb:.4f}")
print(f"ROC-AUC  : {auc_xgb:.4f}")

# ========================================================================
# ETAPE 6 : CATBOOST - OPTIMISATION RAPIDE AVEC OPTUNA
# ========================================================================
print("\n" + "="*70)
print("CATBOOST - Optimisation rapide avec Optuna")
print("="*70)

def objective_cat_fast(trial):
    param = {
        "iterations": trial.suggest_int("iterations", 300, 700),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.05, 0.1]),
        "depth": trial.suggest_int("depth", 3, 5),
        "bootstrap_type": "Bernoulli",
        "subsample": trial.suggest_categorical("subsample", [0.8, 0.9]),
        "colsample_bylevel": trial.suggest_categorical("colsample_bylevel", [0.8, 1.0]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 40),
        "l2_leaf_reg": trial.suggest_categorical("l2_leaf_reg", [1, 3, 5]),
        "random_seed": RSEED,
        "loss_function": "MultiClass",
        "verbose": False,
        "task_type": "CPU",
        "thread_count": 4,
    }

    model = CatBoostClassifier(**param)

    try:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RSEED)
        scores = []
        for tr_idx, vl_idx in cv.split(X_train, y_train):
            X_tr, X_vl = X_train.iloc[tr_idx], X_train.iloc[vl_idx]
            y_tr, y_vl = y_train[tr_idx], y_train[vl_idx]

            model_fold = CatBoostClassifier(**param)
            model_fold.fit(
                X_tr, y_tr,
                cat_features=cat_features_idx,
                eval_set=(X_vl, y_vl),
                early_stopping_rounds=30,
                verbose=False
            )
            y_pred = model_fold.predict(X_vl)
            scores.append(accuracy_score(y_vl, y_pred))

        return np.mean(scores)
    except Exception as e:
        print(f"Trial {trial.number} échoue : {e}")
        return 0.0

study_cat = optuna.create_study(direction="maximize", pruner=MedianPruner())
study_cat.optimize(objective_cat_fast, n_trials=20, show_progress_bar=True)

print("\n" + "="*70)
print("CATBOOST - Meilleurs hyperparamètres")
print("="*70)
for k, v in study_cat.best_params.items():
    print(f"  {k}: {v}")
print(f"\nMeilleur score CV 3-fold : {study_cat.best_value:.4f}")

best_params_cat = study_cat.best_params.copy()

cat_final = CatBoostClassifier(
    loss_function="MultiClass",
    random_seed=RSEED,
    verbose=False,
    task_type="CPU",
    early_stopping_rounds=50,
    bootstrap_type="Bernoulli",
    **best_params_cat
)

print("\nEntraînement du modèle final CatBoost...")
cat_final.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_features_idx,
    use_best_model=True
)

y_val_proba_cat = cat_final.predict_proba(X_val)
y_val_pred_cat = np.argmax(y_val_proba_cat, axis=1)

acc_cat = accuracy_score(y_val, y_val_pred_cat)
f1m_cat = f1_score(y_val, y_val_pred_cat, average="macro")
auc_cat = roc_auc_score(y_val, y_val_proba_cat, multi_class="ovr")

print("\n" + "="*70)
print("CATBOOST - Performance validation")
print("="*70)
print(f"Accuracy  : {acc_cat:.4f}")
print(f"F1-macro : {f1m_cat:.4f}")
print(f"ROC-AUC  : {auc_cat:.4f}")


# ========================================================================
# ETAPE 6bis : RANDOM FOREST - BASELINE ONE-VS-ALL
# ========================================================================
print("\n" + "="*70)
print("RANDOM FOREST - Baseline One-vs-All")
print("="*70)

# Ici j'utilise OneVsRestClassifier autour d'un RandomForest.
# Comme y a 4 classes, One-vs-All entraîne 4 forêts binaires (classe k vs les autres).
rf_base = OneVsRestClassifier(
    RandomForestClassifier(
        n_estimators=300,         # compromis perf / temps
        max_depth=None,           # profondeur libre
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=RSEED,
        n_jobs=-1
    )
)

rf_base.fit(X_train, y_train)
y_val_pred_rf = rf_base.predict(X_val)


acc_rf = accuracy_score(y_val, y_val_pred_rf)
f1m_rf = f1_score(y_val, y_val_pred_rf, average="macro")

print(f"Accuracy  : {acc_rf:.4f}")
print(f"F1-macro : {f1m_rf:.4f}")


# ========================================================================
# ETAPE 7 : STACKING avec meta-LightGBM
# ========================================================================
print("\n" + "="*70)
print("STACKING - Meta-modèle LightGBM")
print("="*70)

print("Génération des prédictions OOF pour le stacking...")
# Préparation des matrices OOF pour chaque modèle 
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RSEED)

y_train_proba_lgb_oof = np.zeros((len(X_train), 4))
y_train_proba_xgb_oof = np.zeros((len(X_train), 4))
y_train_proba_cat_oof = np.zeros((len(X_train), 4))

# Boucle sur les folds
for fold, (tr_idx, vl_idx) in enumerate(cv_outer.split(X_train, y_train)):
    print(f"Fold {fold+1}/5...", end=" ")

    X_tr_lgb, X_vl_lgb = X_train_lgb.iloc[tr_idx], X_train_lgb.iloc[vl_idx]
    X_tr, X_vl = X_train.iloc[tr_idx], X_train.iloc[vl_idx]
    y_tr, y_vl = y_train[tr_idx], y_train[vl_idx]

    # LightGBM
    lgb_cv = LGBMClassifier(
        objective="multiclass",
        num_class=4,
        subsample_freq=1,
        random_state=RSEED,
        n_jobs=-1,
        verbosity=-1,
        **best_params_lgbm
    )
    lgb_cv.fit(
        X_tr_lgb, y_tr,
        categorical_feature=cat_cols,
        callbacks=[log_evaluation(period=0)]
    )
    y_train_proba_lgb_oof[vl_idx] = lgb_cv.predict_proba(X_vl_lgb)

    # XGBoost
    xgb_cv = XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        random_state=RSEED,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
        **best_params_xgb
    )
    xgb_cv.fit(X_tr, y_tr, verbose=False)
    y_train_proba_xgb_oof[vl_idx] = xgb_cv.predict_proba(X_vl)

    # CatBoost
    cat_params_cv = best_params_cat.copy()
    cat_params_cv["bootstrap_type"] = "Bernoulli"
    cat_params_cv["loss_function"] = "MultiClass"
    cat_params_cv["random_seed"] = RSEED
    cat_params_cv["verbose"] = False
    cat_params_cv["task_type"] = "CPU"

    cat_cv = CatBoostClassifier(**cat_params_cv)
    cat_cv.fit(X_tr, y_tr, cat_features=cat_features_idx)
    y_train_proba_cat_oof[vl_idx] = cat_cv.predict_proba(X_vl)

    print("OK")

# Matrices meta train/val 
X_meta_train = np.hstack([
    y_train_proba_lgb_oof,
    y_train_proba_xgb_oof,
    y_train_proba_cat_oof
])

X_meta_val = np.hstack([
    y_val_proba_lgb,
    y_val_proba_xgb,
    y_val_proba_cat
])

# Meta-model : LightGBM simple et régularisé
meta_model = LGBMClassifier(
    objective="multiclass",
    num_class=4,
    n_estimators=400,
    learning_rate=0.03,
    max_depth=3,
    num_leaves=15,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.1,
    random_state=RSEED,
    n_jobs=-1
)

print("\nEntraînement du meta-modèle LightGBM...")
meta_model.fit(
    X_meta_train, y_train,
    eval_set=[(X_meta_val, y_val)],
    callbacks=[early_stopping(stopping_rounds=50),
               log_evaluation(period=0)]
)

y_val_pred_stack = meta_model.predict(X_meta_val)
y_val_proba_stack = meta_model.predict_proba(X_meta_val)

acc_stack = accuracy_score(y_val, y_val_pred_stack)
f1m_stack = f1_score(y_val, y_val_pred_stack, average="macro")
auc_stack = roc_auc_score(y_val, y_val_proba_stack, multi_class="ovr")

print("\n" + "="*70)
print("RESUME DES PERFORMANCES (features V2)")
print("="*70)
print(f"{'Méthode':<30} {'Accuracy':<10} {'F1-macro':<10} {'ROC-AUC':<10}")
print("-"*60)
print(f"{'Random Forest  One-vs-All (baseline)':<30} {acc_rf:<10.4f} {f1m_rf:<10.4f} {'-':<10}")
print(f"{'LightGBM seul':<30} {acc_lgb:<10.4f} {f1m_lgb:<10.4f} {auc_lgb:<10.4f}")
print(f"{'XGBoost seul':<30} {acc_xgb:<10.4f} {f1m_xgb:<10.4f} {auc_xgb:<10.4f}")
print(f"{'CatBoost seul':<30} {acc_cat:<10.4f} {f1m_cat:<10.4f} {auc_cat:<10.4f}")
print(f"{'Stacking meta-LGBM':<30} {acc_stack:<10.4f} {f1m_stack:<10.4f} {auc_stack:<10.4f}")


# ========================================================================
# ETAPE 8 : FEATURE IMPORTANCE MOYENNE
# ========================================================================
print("\n" + "="*70)
print("TOP 30 FEATURES LES PLUS IMPORTANTES (moyenne des 3 modèles)")
print("="*70)

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance_lgb": lgb_final.feature_importances_,
    "importance_xgb": xgb_final.feature_importances_,
    "importance_cat": cat_final.feature_importances_
})
feature_importance["importance_mean"] = (
    feature_importance["importance_lgb"]
    + feature_importance["importance_xgb"]
    + feature_importance["importance_cat"]
) / 3

feature_importance = feature_importance.sort_values("importance_mean", ascending=False)
print(feature_importance.head(30).to_string(index=False))

# ========================================================================
# ETAPE 9 : ANALYSE SHAP - FEATURES ORIGINALES vs V2
# ========================================================================
print("\n" + "="*70)
print("ANALYSE SHAP (LightGBM) - Original vs Features V2")
print("="*70)

# --- a) Modèle LightGBM sur FEATURES ORIGINALES ---
# On prépare X_orig_lgb avec mêmes indices et types
X_orig_lgb = X_orig.copy()
for c in cat_cols:
    if c in X_orig_lgb.columns:
        X_orig_lgb[c] = X_orig_lgb[c].astype("category")

X_train_orig_lgb = X_orig_lgb.iloc[idx_train].reset_index(drop=True)
X_val_orig_lgb = X_orig_lgb.iloc[idx_val].reset_index(drop=True)

lgb_orig = LGBMClassifier(
    objective="multiclass",
    num_class=4,
    subsample_freq=1,
    random_state=RSEED,
    n_jobs=-1,
    verbosity=-1,
    **best_params_lgbm
)

print("\nEntraînement LightGBM sur features ORIGINALES...")
lgb_orig.fit(
    X_train_orig_lgb, y_train,
    eval_set=[(X_val_orig_lgb, y_val)],
    callbacks=[early_stopping(stopping_rounds=50),
               log_evaluation(period=0)],
    categorical_feature=cat_cols
)

# Echantillon pour SHAP
sample_size = min(2000, len(X_val_orig_lgb))
X_shap_orig = X_val_orig_lgb.sample(n=sample_size, random_state=RSEED)

print(f"Calcul des valeurs SHAP (features originales) sur {sample_size} obs...")
explainer_orig = shap.TreeExplainer(lgb_orig)
shap_values_orig = explainer_orig(X_shap_orig)

# On moyenne les SHAP sur les classes (multiclass → (n, p, K))
sv_orig = shap_values_orig.values  # (n, p, K)
sv_orig_mean = sv_orig.mean(axis=2)

plt.figure(figsize=(10, 7))
shap.summary_plot(
    sv_orig_mean,
    X_shap_orig,
    show=False,
    max_display=20
)
plt.tight_layout()
plt.savefig(data_path / "shap_summary_original.png", dpi=300, bbox_inches="tight")
plt.close()


# --- b Modèle LightGBM sur FEATURES V2 (déjà entraîné : lgb_final) ---
sample_size_v2 = min(2000, len(X_val_lgb))
X_shap_v2 = X_val_lgb.sample(n=sample_size_v2, random_state=RSEED)

print(f"\nCalcul des valeurs SHAP (features V2) sur {sample_size_v2} obs...")
explainer_v2 = shap.TreeExplainer(lgb_final)
shap_values_v2 = explainer_v2(X_shap_v2)
sv_v2 = shap_values_v2.values  # (n, p, K)
sv_v2_mean = sv_v2.mean(axis=2)

plt.figure(figsize=(10, 7))
shap.summary_plot(
    sv_v2_mean,
    X_shap_v2,
    show=False,
    max_display=20
)
plt.tight_layout()
plt.savefig(data_path / "shap_summary_v2.png", dpi=300, bbox_inches="tight")
plt.close()
print("   -> shap_summary_v2.png sauvegardé")

# ========================================================================
# ETAPE 10 : PREDICTIONS FINALES SUR TEST AVEC STACKING
# ========================================================================
print("\n" + "="*70)
print("PREDICTIONS FINALES SUR TEST (Stacking)")
print("="*70)

print("\nRéentraînement final des modèles de base sur TOUT le dataset (features V2)...")

# LightGBM
lgb_submit = LGBMClassifier(
    objective="multiclass",
    num_class=4,
    subsample_freq=1,
    random_state=RSEED,
    n_jobs=-1,
    verbosity=-1,
    **best_params_lgbm
)
lgb_submit.fit(
    X_fe_lgb, y_enc,
    categorical_feature=cat_cols,
    callbacks=[log_evaluation(period=0)]
)

# XGBoost
xgb_submit = XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    random_state=RSEED,
    n_jobs=-1,
    tree_method="hist",
    verbosity=0,
    **best_params_xgb
)
xgb_submit.fit(X_fe, y_enc, verbose=False)

# CatBoost
cat_params_submit = best_params_cat.copy()
cat_params_submit["bootstrap_type"] = "Bernoulli"
cat_params_submit["loss_function"] = "MultiClass"
cat_params_submit["random_seed"] = RSEED
cat_params_submit["verbose"] = False
cat_params_submit["task_type"] = "CPU"


cat_submit = CatBoostClassifier(**cat_params_submit)
cat_submit.fit(X_fe, y_enc, cat_features=cat_features_idx)

print("Prédictions de probabilités sur test...")
y_test_proba_lgb = lgb_submit.predict_proba(X_test_fe_lgb)
y_test_proba_xgb = xgb_submit.predict_proba(X_test_fe)
y_test_proba_cat = cat_submit.predict_proba(X_test_fe)

# Features meta pour le test (mêmes blocs que pour X_meta_val)
X_meta_test = np.hstack([
    y_test_proba_lgb,
    y_test_proba_xgb,
    y_test_proba_cat
])

y_test_pred_stack_idx = meta_model.predict(X_meta_test)
y_test_pred_stack_lbl = le.inverse_transform(y_test_pred_stack_idx)

submission_stack = pd.DataFrame({
    "id": test_id.astype(int),
    "y": y_test_pred_stack_lbl
})
submission_stack.to_csv(data_path / "submission_v2_stacking_metaLGBM.csv", index=False)


