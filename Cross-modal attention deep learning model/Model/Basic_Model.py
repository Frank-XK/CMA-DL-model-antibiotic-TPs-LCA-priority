import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
import os
import pickle
from datetime import datetime
from rdkit import DataStructs
from sklearn.base import clone

warnings.filterwarnings("ignore")


# ==================== settings ====================
class Config:
    def __init__(self):
        self.seed = 151
        self.k_folds = 10
        self.num_classes = 3
        self.morgan_radius = 2
        self.morgan_nbits = 1024
        self.random_state = 42
        self.data_path = r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\origion-data2.xlsx"
        self.output_dir = "Basic_model_results"
        self.fingerprints = ['morgan', 'maccs', 'pharm']
        self.classifiers = {
            'RandomForest': RandomForestClassifier(n_estimators=42, max_depth=50,
                                                   class_weight='balanced', random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                                         random_state=self.random_state),
            'SVM': SVC(kernel='linear', C=1.0, probability=True,
                       class_weight='balanced', random_state=self.random_state)
        }


config = Config()

# Create output directory
if not os.path.exists(config.output_dir):
    os.makedirs(config.output_dir)


# ==================== fingerprint ====================
def mol_to_fingerprint(mol, fp_type='morgan'):

    if fp_type == 'morgan':
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        mfpgen = GetMorganGenerator(radius=config.morgan_radius, fpSize=config.morgan_nbits)
        fp = mfpgen.GetFingerprint(mol)
        arr = np.zeros((config.morgan_nbits,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    elif fp_type == 'maccs':
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((167,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    elif fp_type == 'pharm':
        fp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
        return np.array(fp).astype(np.float32)

# ==================== load data ====================

def load_data(filepath):
    df = pd.read_excel(filepath)
    label_map = {'Ⅰ': 0, 'Ⅱ': 1, 'Ⅲ': 2}
    fps = []
    labels = []
    for _, row in df.iterrows():
        smiles = row['SMILES']
        grade = row['Grade']
        if grade not in label_map:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = mol_to_fingerprint(mol, config.fingerprint_type)
        fps.append(fp)
        labels.append(label_map[grade])
    X = np.array(fps)
    y = np.array(labels)
    return X, y


# ==================== evaluation ====================
def compute_metrics(true, pred, probs):
    metrics = {}
    try:
        metrics['auc'] = roc_auc_score(true, probs, multi_class='ovr')
    except Exception:
        metrics['auc'] = float('nan')
    metrics['f1'] = f1_score(true, pred, average='macro')
    metrics['ba'] = balanced_accuracy_score(true, pred)
    metrics['mcc'] = matthews_corrcoef(true, pred)
    return metrics


# ==================== save model ====================
def save_model(model, scaler, fingerprint_type, classifier_name, fold_idx):
    model_dir = os.path.join(config.output_dir, "saved_models", f"{fingerprint_type}_{classifier_name}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"fold{fold_idx}_Basic.pkl")

    save_data = {
        'model': model,
        'scaler': scaler,
        'fingerprint_type': fingerprint_type,
        'classifier_name': classifier_name,
        'fold_idx': fold_idx,
        'timestamp': timestamp
    }

    with open(model_path, 'wb') as f:
        pickle.dump(save_data, f)

    return model_path


# ==================== calculate metrics ====================
def calculate_mean_metrics(metrics_list):
    mean_metrics = {}
    for metric in metrics_list[0].keys():
        values = [m[metric] for m in metrics_list]
        mean_metrics[f'mean_{metric}'] = np.nanmean(values)
        mean_metrics[f'std_{metric}'] = np.nanstd(values)
    return mean_metrics


# ==================== main function ====================
def main():
    print("Loading data...")
    df = pd.read_excel(config.data_path)
    label_map = {'Ⅰ': 0, 'Ⅱ': 1, 'Ⅲ': 2}

    detailed_results = pd.DataFrame(columns=[
        'fingerprint', 'classifier', 'fold', 'auc', 'f1', 'ba', 'mcc'
    ])

    mean_results = pd.DataFrame(columns=[
        'fingerprint', 'classifier', 'mean_auc', 'std_auc',
        'mean_f1', 'std_f1', 'mean_ba', 'std_ba', 'mean_mcc', 'std_mcc'
    ])

    for fp_type in config.fingerprints:
        print(f"\n=== Use fingerprint: {fp_type} ===")

        fps = []
        labels = []
        for _, row in df.iterrows():
            smiles = row['SMILES']
            grade = row['Grade']
            if grade not in label_map:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fp = mol_to_fingerprint(mol, fp_type)
            fps.append(fp)
            labels.append(label_map[grade])
        X = np.array(fps)
        y = np.array(labels)
        print(f"{X.shape[0]}, {X.shape[1]}")

        skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)

        for clf_name, clf in config.classifiers.items():
            print(f"\n=== Using classifier: {clf_name} ===")
            all_metrics = []

            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = None
                if fp_type == 'pharm':
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                current_clf = clone(clf)
                current_clf.fit(X_train, y_train)

                # save model
                model_path = save_model(current_clf, scaler, fp_type, clf_name, fold + 1)

                y_pred = current_clf.predict(X_test)
                y_prob = current_clf.predict_proba(X_test) if hasattr(current_clf, "predict_proba") else None

                if y_prob is None:
                    y_prob = np.zeros((len(y_test), config.num_classes))
                    for i, cls in enumerate(np.unique(y_test)):
                        y_prob[y_pred == cls, i] = 1

                metrics = compute_metrics(y_test, y_pred, y_prob)
                all_metrics.append(metrics)

                print(f"Fold {fold + 1} - AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}, "
                      f"BA: {metrics['ba']:.4f}, MCC: {metrics['mcc']:.4f}")

                detailed_results.loc[len(detailed_results)] = {
                    'fingerprint': fp_type,
                    'classifier': clf_name,
                    'fold': fold + 1,
                    'auc': metrics['auc'],
                    'f1': metrics['f1'],
                    'ba': metrics['ba'],
                    'mcc': metrics['mcc']
                }

            mean_metrics = calculate_mean_metrics(all_metrics)
            mean_results.loc[len(mean_results)] = {
                'fingerprint': fp_type,
                'classifier': clf_name,
                **mean_metrics
            }

            print(f"\n{fp_type} + {clf_name} 10-fold cross-validation average index:")
            print(f"AUC: {mean_metrics['mean_auc']:.4f}  ± {mean_metrics['std_auc']:.4f}")
            print(f"F1: {mean_metrics['mean_f1']:.4f}  ± {mean_metrics['std_f1']:.4f}")
            print(f"BA: {mean_metrics['mean_ba']:.4f}  ± {mean_metrics['std_ba']:.4f}")
            print(f"MCC: {mean_metrics['mean_mcc']:.4f}  ± {mean_metrics['std_mcc']:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Detailed results
    detailed_excel_path = os.path.join(config.output_dir, f"detailed_metrics_Basic.xlsx")
    detailed_results.to_excel(detailed_excel_path, index=False)

    # Average results
    mean_excel_path = os.path.join(config.output_dir, f"mean_metrics_Basic.xlsx")
    mean_results.to_excel(mean_excel_path, index=False)

    print(f"Detailed evaluation results have been saved to: {detailed_excel_path}")
    print(f"Average evaluation results have been saved to: {mean_excel_path}")


if __name__ == "__main__":
    main()