
import os
import warnings
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn import GINConv, AvgPooling

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score, matthews_corrcoef,
    roc_curve, precision_recall_curve, confusion_matrix, auc, average_precision_score
)

from torch.utils.data import Dataset, DataLoader
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer


# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # Proper handling of minus signs
# ---------------- Global settings ----------------
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Parameter ====================
class Config:
    def __init__(self):
        # experiment
        self.seed = 151
        self.k_folds = 10
        self.epochs = 100
        self.num_classes = 2

        # data
        self.batch_size = 32
        self.num_workers = 0
        self.add_edge_feats = True

        # fingerprint
        self.morgan_radius = 2
        self.morgan_nbits = 1024
        self.maccs_bits = 167
        self.erg_length = 441

        # model
        self.hidden_dim = 128
        self.dropout = 0.23
        self.gnn_num_layers = 4

        # train
        self.lr = 0.00015
        self.best_model_save_path = './best_models-EcoR/'
        self.visualization_path = './visualization_results-EcoR/'

        # path
        self.data_path = r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\origion-data21.xlsx"

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

config = Config()

# ==================== utility functions ====================
def set_seed(seed=config.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def move_batch_to_device(graphs, labels, fps, device):
    graphs = graphs.to(device)
    labels = labels.to(device)
    node_feats = graphs.ndata['h'].to(device)
    edge_feats = graphs.edata['e'].to(device) if 'e' in graphs.edata else None
    fps = {k: v.to(device) for k, v in fps.items()}
    return graphs, labels, node_feats, fps, edge_feats

set_seed()

# ==================== Data processing ====================
def mol_to_fingerprints(mol, morgan_radius=config.morgan_radius, morgan_nbits=config.morgan_nbits):
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=morgan_radius, nBits=morgan_nbits)
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    pharm_fp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
    return {
        'morgan': torch.tensor(morgan_fp, dtype=torch.float32),
        'maccs': torch.tensor(maccs_fp, dtype=torch.float32),
        'pharm': torch.tensor(pharm_fp, dtype=torch.float32)
    }

class MolecularDataset(Dataset):
    def __init__(self, df, add_edge_feats=config.add_edge_feats,
                 morgan_radius=config.morgan_radius, morgan_nbits=config.morgan_nbits):
        self.labels = {'Ⅰ': 0, 'Ⅱ': 1}
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.add_edge_feats = add_edge_feats
        if self.add_edge_feats:
            self.bond_featurizer = CanonicalBondFeaturizer()
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits

        self.data = []
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol is not None and row['EcoR'] in self.labels:
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        smiles = row['SMILES']
        label = self.labels[row['EcoR']]
        mol = Chem.MolFromSmiles(smiles)

        if self.add_edge_feats:
            g = smiles_to_bigraph(smiles, node_featurizer=self.atom_featurizer,
                                  edge_featurizer=self.bond_featurizer)
        else:
            g = smiles_to_bigraph(smiles, node_featurizer=self.atom_featurizer)

        fps = mol_to_fingerprints(mol, self.morgan_radius, self.morgan_nbits)
        return g, label, fps

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None, None
    graphs, labels, fps_list = zip(*batch)
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    fp_batch = {k: torch.stack([fps[k] for fps in fps_list], dim=0) for k in fps_list[0].keys()}
    return batched_graph, labels, fp_batch

# ==================== GIN encoding + fingerprint encoding + cross-modal fusion ====================
class FingerprintEncoder(nn.Module):
    def __init__(self, fp_dims, emb_dim=config.hidden_dim, dropout=config.dropout):
        super().__init__()
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for name, dim in fp_dims.items()
        })
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(len(fp_dims) - 1)
        ])

    def forward(self, fps):
        embs = [self.encoders[name](fps[name]) for name in self.encoders.keys()]
        fused = embs[0]
        for i in range(1, len(embs)):
            fused = self.fusion_layers[i - 1](torch.cat([fused, embs[i]], dim=1))
        return fused

class GIN_Encoder(nn.Module):
    def __init__(self, in_feats, hidden_dim=config.hidden_dim, num_layers=config.gnn_num_layers, dropout=config.dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_residual = []
        self.input_norm = nn.LayerNorm(in_feats)
        for i in range(num_layers):
            in_dim = in_feats if i == 0 else hidden_dim
            out_dim = hidden_dim
            self.use_residual.append(i > 0)
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU()
            )
            self.layers.append(GINConv(mlp, 'sum'))
            self.add_module(f"norm_{i}", nn.LayerNorm(out_dim))
            self.add_module(f"dropout_{i}", nn.Dropout(dropout))
        self.pool = AvgPooling()

    def forward(self, g, h):
        h = self.input_norm(h)
        for i, layer in enumerate(self.layers):
            residual = h if self.use_residual[i] else 0
            h = layer(g, h)
            h = getattr(self, f"norm_{i}")(h)
            h = F.relu(h)
            if self.use_residual[i]:
                h = h + residual
            h = getattr(self, f"dropout_{i}")(h)
        return self.pool(g, h)

class CrossModalFusionEncoder(nn.Module):
    def __init__(self, node_feat_dim, fp_dims, hidden_dim=config.hidden_dim):
        super().__init__()
        self.graph_encoder = GIN_Encoder(node_feat_dim, hidden_dim)
        self.fp_encoder = FingerprintEncoder(fp_dims, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, dropout=config.dropout, batch_first=False)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, g, h, fps):
        graph_feat = self.graph_encoder(g, h)               # [B, H]
        fp_feat = self.fp_encoder(fps)                      # [B, H]

        fp_attn, _ = self.cross_attn(graph_feat.unsqueeze(0), fp_feat.unsqueeze(0), fp_feat.unsqueeze(0))
        fused = self.fusion(torch.cat([graph_feat, fp_attn.squeeze(0)], dim=1))
        return fused

class MoleculeModel(nn.Module):

    def __init__(self, node_feat_dim, fp_dims, hidden_dim=config.hidden_dim, num_classes=config.num_classes):
        super().__init__()
        self.encoder = CrossModalFusionEncoder(node_feat_dim, fp_dims, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, g, node_feats, fps, edge_feats=None):
        x = self.encoder(g, node_feats, fps)
        return self.classifier(x)

# ==================== Training/Assessment ====================
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for graphs, labels, fps in train_loader:
        if graphs is None:
            continue
        graphs, labels, node_feats, fps, edge_feats = move_batch_to_device(graphs, labels, fps, device)
        optimizer.zero_grad()
        outputs = model(graphs, node_feats, fps, edge_feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(train_loader))

def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for graphs, labels, fps in loader:
            if graphs is None:
                continue
            graphs, labels, node_feats, fps, edge_feats = move_batch_to_device(graphs, labels, fps, device)
            outputs = model(graphs, node_feats, fps, edge_feats)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds, all_probs

def compute_metrics(true, pred, probs):
    metrics = {}
    true = np.asarray(true)
    pred = np.asarray(pred)
    probs = np.asarray(probs)


    pos_scores = probs[:, 1] if probs.ndim == 2 and probs.shape[1] == 2 else probs
    metrics['auc'] = roc_auc_score(true, pos_scores)
    metrics['f1']  = f1_score(true, pred, average='binary', pos_label=1)
    metrics['ba']  = balanced_accuracy_score(true, pred)
    metrics['mcc'] = matthews_corrcoef(true, pred)
    return metrics

def select_best_fold_lexi(fold_metrics, primary='auc', tie_breakers=('mcc','ba','f1')):

    metrics_matrix = []
    for m in fold_metrics:
        row = [m.get(primary, float('nan'))] + [m.get(k, float('nan')) for k in tie_breakers]
        metrics_matrix.append(row)
    metrics_matrix = np.asarray(metrics_matrix)


    keys = [metrics_matrix[:, i] for i in range(metrics_matrix.shape[1]-1, -1, -1)]
    order = np.lexsort(keys)
    best_idx = order[-1]
    return int(best_idx)


# ==================== Training/Assessment ====================

def _is_cv_list(x):

    return isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple, np.ndarray))


def plot_combined_curves(true_labels, pred_probs, model_names, num_classes=2, save_path='./results'):

    ensure_dir(save_path)
    mean_fpr    = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)

    plt.figure(figsize=(14, 6))

    # ---------- ROC ----------
    plt.subplot(121)
    for model_idx, name in enumerate(model_names):
        ys = true_labels[model_idx]
        ps = pred_probs[model_idx]


        if _is_cv_list(ys) and _is_cv_list(ps):
            folds_true = ys
            folds_prob = ps
        else:
            folds_true = [np.asarray(ys)]
            folds_prob = [np.asarray(ps)]

        tprs, aucs = [], []
        for y, prob in zip(folds_true, folds_prob):
            y = np.asarray(y); prob = np.asarray(prob)
            scores = prob[:, 1]
            fpr, tpr, _ = roc_curve(y, scores)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = float(np.mean(aucs))
        plt.plot(mean_fpr, mean_tpr, lw=2.5, label=f'{name} AUC = {mean_auc:.3f}')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve')
    plt.legend(loc='lower right', fontsize=11, frameon=True)

    # ---------- PR ----------
    plt.subplot(122)
    for model_idx, name in enumerate(model_names):
        ys = true_labels[model_idx]
        ps = pred_probs[model_idx]

        if _is_cv_list(ys) and _is_cv_list(ps):
            folds_true = ys
            folds_prob = ps
        else:
            folds_true = [np.asarray(ys)]
            folds_prob = [np.asarray(ps)]

        precisions, aps = [], []
        for y, prob in zip(folds_true, folds_prob):
            y = np.asarray(y); prob = np.asarray(prob)
            scores = prob[:, 1]
            precision, recall, _ = precision_recall_curve(y, scores)

            interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
            precisions.append(interp_precision)
            aps.append(average_precision_score(y, scores))

        mean_precision = np.mean(precisions, axis=0)
        mean_ap = float(np.mean(aps))
        plt.plot(mean_recall, mean_precision, lw=2.5, label=f'{name} AP = {mean_ap:.3f}')

    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left', fontsize=11, frameon=True)

    plt.tight_layout()
    out_path = os.path.join(save_path, 'performance_curves_binary.png')
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(true_labels, pred_labels, model_names,
                            num_classes=2, save_path='./results', normalize_confusion=False):

    ensure_dir(save_path)
    xticks = yticks = ['Ⅰ (0)', 'Ⅱ (1)']

    def _is_cv_list(x):
        return isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple, np.ndarray))

    for model_idx, name in enumerate(model_names):
        ys = true_labels[model_idx]
        ps = pred_labels[model_idx]

        if _is_cv_list(ys) and _is_cv_list(ps):
            folds_true = ys
            folds_pred = ps
        else:
            folds_true = [np.asarray(ys)]
            folds_pred = [np.asarray(ps)]


        cm_sum = np.zeros((2, 2), dtype=np.int64)
        for y, p in zip(folds_true, folds_pred):
            y = np.asarray(y).astype(int)
            p = np.asarray(p).astype(int)
            cm = confusion_matrix(y, p, labels=[0, 1]).astype(np.int64)
            cm_sum += cm

        if normalize_confusion:

            cm_plot = cm_sum.astype(np.float64)
            row_sums = cm_plot.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_plot = cm_plot / row_sums
            fmt = '.2f'
        else:

            cm_plot = cm_sum  # int64
            fmt = 'd'

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap='Reds',
                    xticklabels=xticks, yticklabels=yticks)
        plt.title(f'Confusion Matrix - EcoR')
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.tight_layout()
        out_path = os.path.join(save_path, f'confusion_matrix_{name}.png')
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close()

# ==================== Cross-validation training ====================
def train_and_evaluate_fusion_gin(dataset, labels, in_feats, fp_dims, save_path):
    ensure_dir(save_path)
    skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)

    best_models = []
    fold_metrics = []
    all_metrics = {'auc': [], 'f1': [], 'ba': [], 'mcc': []}

    all_folds_data = {'true_labels': [], 'pred_labels': [], 'pred_probs': [], 'test_indices': []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
        print(f"\n=== FUSION + GIN Fold {fold + 1}/{config.k_folds} ===")

        train_loader = DataLoader([dataset[i] for i in train_idx],
                                  batch_size=config.batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=config.num_workers)
        test_loader  = DataLoader([dataset[i] for i in test_idx],
                                  batch_size=config.batch_size, shuffle=False,
                                  collate_fn=collate_fn, num_workers=config.num_workers)

        model = MoleculeModel(node_feat_dim=in_feats, fp_dims=fp_dims,
                              hidden_dim=config.hidden_dim, num_classes=config.num_classes).to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss()

        best_auc = -1.0
        best_model_state, best_fold_data, current_metrics = None, None, None

        for epoch in range(config.epochs):
            loss_val = train(model, train_loader, optimizer, criterion)

            if (epoch + 1) % 10 == 0 or epoch == config.epochs - 1:
                model.eval()
                all_true, all_pred, all_probs = [], [], []
                with torch.no_grad():
                    for graphs, batch_labels, fps in test_loader:
                        if graphs is None:
                            continue
                        graphs, batch_labels, node_feats, fps, edge_feats = move_batch_to_device(graphs, batch_labels, fps, device)
                        outputs = model(graphs, node_feats, fps, edge_feats)
                        probs = F.softmax(outputs, dim=1)
                        _, preds = torch.max(outputs, 1)
                        all_true.extend(batch_labels.cpu().numpy())
                        all_pred.extend(preds.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())

                metrics = compute_metrics(all_true, all_pred, all_probs)
                print(f"Epoch {epoch + 1}/{config.epochs}  Loss: {loss_val:.4f}  AUC: {metrics['auc']:.4f}")

                if metrics['auc'] > best_auc:
                    best_auc = metrics['auc']
                    best_model_state = deepcopy(model.state_dict())
                    current_metrics = metrics
                    best_fold_data = {
                        'true_labels': np.array(all_true),
                        'pred_labels': np.array(all_pred),
                        'pred_probs': np.array(all_probs),
                        'test_indices': test_idx
                    }


        if best_model_state is not None:
            model_path = os.path.join(save_path, f'best_model_fold_{fold + 1}.pth')
            torch.save({
                'model_state': best_model_state,
                'fold': fold + 1,
                'metrics': current_metrics,
                'config': {'model': 'fusion+gin'},
                'plot_data': best_fold_data
            }, model_path)

            np.savez(os.path.join(save_path, f'plot_data_fold_{fold + 1}.npz'), **best_fold_data)

            best_models.append(model_path)
            fold_metrics.append(current_metrics)
            for k in all_metrics: all_metrics[k].append(current_metrics[k])

            all_folds_data['true_labels'].append(best_fold_data['true_labels'])
            all_folds_data['pred_labels'].append(best_fold_data['pred_labels'])
            all_folds_data['pred_probs'].append(best_fold_data['pred_probs'])
            all_folds_data['test_indices'].append(best_fold_data['test_indices'])

            print(f"\nFold {fold + 1} Best  AUC: {current_metrics['auc']:.4f}  F1: {current_metrics['f1']:.4f}  "
                  f"BA: {current_metrics['ba']:.4f}  MCC: {current_metrics['mcc']:.4f}")

    final_plot_data = {
        'true_labels' : np.concatenate(all_folds_data['true_labels']),
        'pred_labels' : np.concatenate(all_folds_data['pred_labels']),
        'pred_probs'  : np.vstack(all_folds_data['pred_probs']),
        'test_indices': np.concatenate(all_folds_data['test_indices'])
    }

    print(f"\n=== FUSION + GIN CV Results ===")
    print("Test Metrics (Mean ± SD):")
    print(f"AUC: {np.nanmean(all_metrics['auc']):.4f}  ± {np.nanstd(all_metrics['auc']):.4f}")
    print(f"F1 : {np.mean(all_metrics['f1']):.4f}   ± {np.std(all_metrics['f1']):.4f}")
    print(f"BA : {np.mean(all_metrics['ba']):.4f}   ± {np.std(all_metrics['ba']):.4f}")
    print(f"MCC: {np.mean(all_metrics['mcc']):.4f}  ± {np.nanstd(all_metrics['mcc']):.4f}")

    return best_models, fold_metrics, all_metrics, final_plot_data

# ==================== metrics ====================
def export_metrics_to_excel(all_results, save_path):
    all_dfs = []
    for model_name, result in all_results.items():
        df = pd.DataFrame(result['fold_metrics'])
        df['model_name'] = model_name
        df['fold'] = range(1, len(df) + 1)
        all_dfs.append(df)
    final_df = pd.concat(all_dfs, ignore_index=True)
    summary_df = final_df.groupby('model_name').agg(['mean', 'std'])
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    summary_df.reset_index(inplace=True)
    with pd.ExcelWriter(save_path) as writer:
        final_df.to_excel(writer, sheet_name='Fold_Metrics', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    print(f"Model metrics {save_path}")

# ==================== Main process ====================
def main():
    os.makedirs(config.visualization_path,  exist_ok=True)
    os.makedirs(config.best_model_save_path, exist_ok=True)
    set_seed(config.seed)

    print("Loading data...")
    df = pd.read_excel(config.data_path)
    dataset = MolecularDataset(df, add_edge_feats=config.add_edge_feats,
                               morgan_radius=config.morgan_radius,
                               morgan_nbits=config.morgan_nbits)

    labels = [dataset.labels[row['EcoR']] for row in dataset.data]
    in_feats = dataset[0][0].ndata['h'].shape[1]
    fp_dims = {'morgan': config.morgan_nbits, 'maccs': config.maccs_bits, 'pharm': config.erg_length}


    name = 'fusion_gin'
    save_path = os.path.join(config.best_model_save_path, name)
    ensure_dir(save_path)

    best_models, fold_metrics, all_metrics, plot_data = train_and_evaluate_fusion_gin(
        dataset=dataset,
        labels=labels,
        in_feats=in_feats,
        fp_dims=fp_dims,
        save_path=save_path
    )

    all_results = {
        name: {
            'best_models': best_models,
            'fold_metrics': fold_metrics,
            'all_metrics': all_metrics,
            'plot_data': plot_data
        }
    }

    # Visualization
    print("\nGenerating visualizations...")
    plot_combined_curves([plot_data['true_labels']], [plot_data['pred_probs']], [name],
                         num_classes=config.num_classes, save_path=config.visualization_path)
    plot_confusion_matrices([plot_data['true_labels']], [plot_data['pred_labels']], [name],
                            num_classes=config.num_classes, save_path=config.visualization_path)

    # Excel
    export_metrics_to_excel(all_results, './model_metrics_summary-EcoR.xlsx')

    print(f"\nTraining complete! Results saved to: {config.visualization_path}")



    best_fold_idx = select_best_fold_lexi(fold_metrics, primary='auc', tie_breakers=('mcc', 'ba', 'f1'))
    best_fold_path = best_models[best_fold_idx]

    ckpt = torch.load(best_fold_path, map_location='cpu')
    unified_best_path = os.path.join(config.best_model_save_path, 'fusion_gin_best_overall-EcoR.pth')

    torch.save({
        'model_state': ckpt['model_state'],
        'best_fold': best_fold_idx + 1,
        'metrics': ckpt.get('metrics', None),
        'model_card': {
            'arch': 'fusion+gin',
            'hidden_dim': config.hidden_dim,
            'num_classes': config.num_classes,
            'gnn_num_layers': config.gnn_num_layers,
            'dropout': config.dropout,
            'fp_dims': {'morgan': config.morgan_nbits, 'maccs': config.maccs_bits, 'pharm': config.erg_length},
            'node_feat_dim': in_feats,
        }
    }, unified_best_path)

    print(f"\n[Global Best] fold = {best_fold_idx + 1}, metrics = {fold_metrics[best_fold_idx]}")
    print(f"Saved unified best model to: {unified_best_path}")


if __name__ == "__main__":

    config.update(
        seed=151,
        epochs=100,
        lr=0.0002,
        dropout=0.23,
        hidden_dim=128,
        batch_size=32,
        k_folds=10,
        num_workers=0,
        morgan_radius=2,
        morgan_nbits=1024,
        maccs_bits=167,
        erg_length=441,
        num_classes=2,
        best_model_save_path='./best_models-EcoR/',
        visualization_path='./visualization_results-EcoR/',
        data_path=r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\origion-data21.xlsx"
    )
    main()
