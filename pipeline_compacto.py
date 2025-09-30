"""Pipeline compacto para treino do ensemble final e geração de submissão.
Regras: Apenas numpy, pandas, scikit-learn.
Para detalhes de EDA e experimentos, ver notebook completo.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('.')
TRAIN_PATH = DATA_DIR / 'train.csv'
TEST_PATH = DATA_DIR / 'test.csv'
SUB_PATH = DATA_DIR / 'sample_submission.csv'
RANDOM_STATE = 42
N_FOLDS = 7

train = pd.read_csv(TRAIN_PATH)
X_test = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SUB_PATH)

target_col = 'target' if 'target' in train.columns else train.columns[-1]
y = train[target_col].values
X = train.drop(columns=[target_col]).copy()

# Detect types (heurística rápida)
cat_cols = [c for c in X.columns if X[c].dtype == 'object' or X[c].nunique() < 20]
num_cols = [c for c in X.columns if c not in cat_cols]
print(f"Cols categóricas: {len(cat_cols)} | numéricas: {len(num_cols)}")

# Encoding
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_enc = X.copy(); X_test_enc = X_test.copy()
if cat_cols:
    X_enc[cat_cols] = encoder.fit_transform(X_enc[cat_cols])
    X_test_enc[cat_cols] = encoder.transform(X_test_enc[cat_cols])
print('Encoding concluído.')

# Remoção baixa variância
low_var_cols = [c for c in X_enc.columns if X_enc[c].nunique() <= 1]
if low_var_cols:
    X_enc.drop(columns=low_var_cols, inplace=True)
    X_test_enc.drop(columns=[c for c in low_var_cols if c in X_test_enc.columns], inplace=True)
print('Cols baixa variância removidas:', low_var_cols)

# Seleção de features via ExtraTrees
from sklearn.ensemble import ExtraTreesClassifier

def select_features_via_importance(X_mat, y, keep_pct=0.9, random_state=42):
    et = ExtraTreesClassifier(n_estimators=400, random_state=random_state, n_jobs=-1)
    et.fit(X_mat, y)
    importances = pd.Series(et.feature_importances_, index=X_mat.columns).sort_values(ascending=False)
    k = max(5, int(len(importances) * keep_pct))
    selected = importances.index[:k].tolist()
    return selected, importances

selected_features, importances = select_features_via_importance(X_enc, y, keep_pct=0.9, random_state=RANDOM_STATE)
print('Features selecionadas:', len(selected_features))
X_sel = X_enc[selected_features]
X_test_sel = X_test_enc[selected_features]

# Modelos base
rf = RandomForestClassifier(n_estimators=800, random_state=RANDOM_STATE, n_jobs=-1)
et = ExtraTreesClassifier(n_estimators=1000, random_state=RANDOM_STATE, n_jobs=-1)
hgb = HistGradientBoostingClassifier(learning_rate=0.06, random_state=RANDOM_STATE)

# CV + blending
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros(len(X_sel))
weights_grid = [(0.33,0.33,0.34),(0.4,0.3,0.3),(0.5,0.25,0.25),(0.34,0.4,0.26),(0.37,0.33,0.30)]
fold_metrics = []
best_acc = -1
best_w = None
test_preds_collect = []

for fold,(tr_idx,val_idx) in enumerate(skf.split(X_sel, y),1):
    X_tr, X_val = X_sel.iloc[tr_idx], X_sel.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    rf.fit(X_tr, y_tr)
    et.fit(X_tr, y_tr)
    hgb.fit(X_tr, y_tr)
    prf = rf.predict_proba(X_val)[:,1]
    pet = et.predict_proba(X_val)[:,1]
    phg = hgb.predict_proba(X_val)[:,1]
    local_best = -1; local_w=None; local_pred=None
    for wr,we,wh in weights_grid:
        blend = wr*prf + we*pet + wh*phg
        pred_label = (blend >= 0.5).astype(int)
        acc = accuracy_score(y_val, pred_label)
        if acc > local_best:
            local_best = acc; local_w=(wr,we,wh); local_pred=blend
    oof_preds[val_idx] = local_pred
    fold_metrics.append(local_best)
    if local_best > best_acc:
        best_acc = local_best; best_w = local_w
    # test predictions weighted by local_w
    prf_t = rf.predict_proba(X_test_sel)[:,1]
    pet_t = et.predict_proba(X_test_sel)[:,1]
    phg_t = hgb.predict_proba(X_test_sel)[:,1]
    test_preds_collect.append(local_w[0]*prf_t + local_w[1]*pet_t + local_w[2]*phg_t)

print('Melhor peso global:', best_w)
print('Acurácias por fold:', np.round(fold_metrics,4))

# Métricas globais OOF baseline threshold 0.5
oof_labels_05 = (oof_preds >= 0.5).astype(int)
cv_accuracy = accuracy_score(y, oof_labels_05)
cv_precision = precision_score(y, oof_labels_05)
cv_recall = recall_score(y, oof_labels_05)
cv_f1 = f1_score(y, oof_labels_05)
print(f'OOF(0.5) Acc={cv_accuracy:.4f} Prec={cv_precision:.4f} Rec={cv_recall:.4f} F1={cv_f1:.4f}')

# Threshold tuning simples
best_thr=0.5; best_thr_acc=-1
for thr in np.linspace(0.3,0.7,21):
    acc = accuracy_score(y, (oof_preds>=thr).astype(int))
    if acc > best_thr_acc:
        best_thr_acc=acc; best_thr=thr
print(f'Melhor threshold={best_thr:.3f} | Acc={best_thr_acc:.4f}')

# Predição final
test_proba = np.mean(test_preds_collect, axis=0)
final_label = (test_proba >= best_thr).astype(int)
submission[target_col] = final_label
out_name = f'submission_compacto_thr{best_thr:.3f}.csv'
submission.to_csv(out_name, index=False)
print('Submissão escrita em', out_name)
