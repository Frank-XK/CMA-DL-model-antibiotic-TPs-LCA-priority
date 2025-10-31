import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc, average_precision_score
from sklearn.preprocessing import label_binarize


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GINConv, AvgPooling, GATConv
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, matthews_corrcoef
from torch.utils.data import Dataset, DataLoader
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import warnings
import random
import os
from copy import deepcopy


warnings.filterwarnings("ignore")

# ==================== Parameter settings ====================
class Config:


    def __init__(self):

        self.seed = 151
        self.model_mode = 'graph'  # Model Pattern: 'fusion'/'graph'/'fp'
        self.num_classes = 3
        self.k_folds = 10
        self.epochs = 500

        self.batch_size = 32
        self.num_workers = 0 
        self.add_edge_feats = True

        self.morgan_radius = 2
        self.morgan_nbits = 1024
        self.maccs_bits = 167
        self.erg_length = 441

        self.hidden_dim = 128
        self.dropout = 0.1
        self.gnn_num_layers = 4

        self.lr = 0.0001
        self.best_model_save_path = './best_models/'
        self.visualization_path = './visualization_results/'

        self.graph_encoder_type = 'gat'   # 'gin', 'gin-eps', 'gat'
        self.gat_num_heads = 4

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Data processing module ====================
def mol_to_fingerprints(mol, morgan_radius=config.morgan_radius,
                        morgan_nbits=config.morgan_nbits):

    # Morgan(ECFP)
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=morgan_radius, nBits=morgan_nbits)

    # MACCS
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)

    # ErG
    pharm_fp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3,
                                         maxPath=21, minPath=1)

    return {
        'morgan': torch.tensor(morgan_fp, dtype=torch.float32),
        'maccs': torch.tensor(maccs_fp, dtype=torch.float32),
        'pharm': torch.tensor(pharm_fp, dtype=torch.float32)
    }

class MolecularDataset(Dataset):
    def __init__(self, df, add_edge_feats=config.add_edge_feats,
                 morgan_radius=config.morgan_radius,
                 morgan_nbits=config.morgan_nbits):
        self.labels = {'Ⅰ': 0, 'Ⅱ': 1, 'Ⅲ': 2} 
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.add_edge_feats = add_edge_feats
        if self.add_edge_feats:
            self.bond_featurizer = CanonicalBondFeaturizer()
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits

        self.data = []
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol is not None and row['Grade'] in self.labels:
                self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data[idx]
        smiles = row['SMILES']
        label = self.labels[row['Grade']]
        mol = Chem.MolFromSmiles(smiles)

        if self.add_edge_feats:
            g = smiles_to_bigraph(smiles,
                                  node_featurizer=self.atom_featurizer,
                                  edge_featurizer=self.bond_featurizer)
        else:
            g = smiles_to_bigraph(smiles,
                                  node_featurizer=self.atom_featurizer)

        atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        g.ndata['atomic_num'] = torch.tensor(atomic_nums, dtype=torch.long)


        fps = mol_to_fingerprints(mol, self.morgan_radius, self.morgan_nbits)

        return g, label, fps

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None, None

    graphs, labels, fps_list = zip(*batch)
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    fp_batch = {}
    for key in fps_list[0].keys():
        fp_batch[key] = torch.stack([fps[key] for fps in fps_list], dim=0)
    return batched_graph, labels, fp_batch


class FingerprintEncoder(nn.Module):
    def __init__(self, fp_dims, emb_dim=config.hidden_dim,
                 dropout=config.dropout):
        super().__init__()
        self.encoders = nn.ModuleDict()
        # Create an independent encoder for each fingerprint
        for name, dim in fp_dims.items():
            self.encoders[name] = nn.Sequential(
                nn.Linear(dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        # The hierarchical fusion middle layer integrates multiple fingerprint codes
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(len(fp_dims) - 1)
        ])

    def forward(self, fps):
        embs = [encoder(fps[name]) for name, encoder in self.encoders.items()]
        fused = embs[0]
        for i in range(1, len(embs)):
            combined = torch.cat([fused, embs[i]], dim=1)
            fused = self.fusion_layers[i - 1](combined)
        return fused

# ==================== GIN and GAT encoders ====================

class GIN_Encoder(nn.Module):
    def __init__(self, in_feats, hidden_dim=config.hidden_dim,
                 num_layers=config.gnn_num_layers):
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
            self.add_module(f"dropout_{i}", nn.Dropout(config.dropout))
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



class GAT_Encoder(nn.Module):
    def __init__(self, in_feats, hidden_dim=128, num_layers=4, gat_num_heads=4, dropout=0.23):
        super(GAT_Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.gat_heads = gat_num_heads
        self.input_norm = nn.LayerNorm(in_feats)

        # GATConv
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        for i in range(num_layers):
            in_dim = in_feats if i == 0 else hidden_dim
            out_dim = hidden_dim
            assert out_dim % self.gat_heads == 0, f"hidden_dim {out_dim} must be divisible by num_heads {self.gat_heads}"
            head_dim = out_dim // self.gat_heads

            conv = GATConv(in_dim, head_dim, num_heads=self.gat_heads,
                           feat_drop=dropout, attn_drop=dropout, residual=False)
            self.layers.append(conv)
            self.norms.append(nn.LayerNorm(out_dim))
            self.dropouts.append(nn.Dropout(dropout))

            if in_dim != out_dim:
                self.residual_projections.append(nn.Linear(in_dim, out_dim))
            else:
                self.residual_projections.append(nn.Identity())

        self.pool = dgl.nn.pytorch.glob.AvgPooling()

    def forward(self, g, h):
        h = self.input_norm(h)
        for i in range(self.num_layers):
            residual = self.residual_projections[i](h)
            h = self.layers[i](g, h)

            h = h.view(h.size(0), -1, self.hidden_dim).squeeze(1)
            h = self.norms[i](h)
            h = F.relu(h)
            h = h + residual
            h = self.dropouts[i](h)
        return self.pool(g, h)


# ==================== CrossModalFusionEncoder and MoleculeModel ====================
class CrossModalFusionEncoder(nn.Module):
    def __init__(self, node_feat_dim, fp_dims, hidden_dim=config.hidden_dim, gnn_type='gin', gat_num_heads=4):
        super().__init__()
        # Choose the appropriate graph encoder based on gnn_type.
        if gnn_type == 'gin':
            self.graph_encoder = GIN_Encoder(node_feat_dim, hidden_dim)
        elif gnn_type == 'gat':
            self.graph_encoder = GAT_Encoder(node_feat_dim, hidden_dim, gat_num_heads=gat_num_heads)
        else:
            raise ValueError(f"Unknown GNN type {gnn_type}")

        self.fp_encoder = FingerprintEncoder(fp_dims, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, dropout=config.dropout)
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
        graph_feat = self.graph_encoder(g, h)  # Graphic features
        fp_feat = self.fp_encoder(fps)  # fingerprint features
        # Cross-modal attention mechanism
        fp_attn, _ = self.cross_attn(query=graph_feat.unsqueeze(0), key=fp_feat.unsqueeze(0), value=fp_feat.unsqueeze(0))
        fused = self.fusion(torch.cat([graph_feat, fp_attn.squeeze(0)], dim=1))
        return fused

class MoleculeModel(nn.Module):
    def __init__(self, node_feat_dim, fp_dims, model_mode=config.model_mode, hidden_dim=config.hidden_dim, num_classes=config.num_classes, gnn_type='gin', gat_num_heads=4):
        super().__init__()
        self.model_mode = model_mode
        if model_mode == 'fusion':
            self.encoder = CrossModalFusionEncoder(node_feat_dim, fp_dims, hidden_dim, gnn_type=gnn_type, gat_num_heads=gat_num_heads)
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
        elif model_mode == 'graph':
            if gnn_type == 'gin':
                self.graph_encoder = GIN_Encoder(node_feat_dim, hidden_dim)
            elif gnn_type == 'gat':
                self.graph_encoder = GAT_Encoder(node_feat_dim, hidden_dim, gat_num_heads=gat_num_heads)
            else:
                raise ValueError(f"Unknown GNN type {gnn_type}")
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        elif model_mode == 'fp':
            self.fp_encoder = FingerprintEncoder(fp_dims, emb_dim=hidden_dim)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            raise ValueError("model_mode must be one of ['fusion', 'graph', 'fp']")

    def forward(self, g, node_feats, fps, edge_feats=None):
        if self.model_mode == 'fusion':
            x = self.encoder(g, node_feats, fps)
        elif self.model_mode == 'graph':
            x = self.graph_encoder(g, node_feats)
        elif self.model_mode == 'fp':
            x = self.fp_encoder(fps)
        return self.classifier(x)



# ==================== Plotting functions ====================

def plot_combined_curves(true_labels, pred_probs, model_names, num_classes=3, save_path='./results'):

    plt.figure(figsize=(14, 6))

    # ROC
    plt.subplot(121)
    for true, probs, name in zip(true_labels, pred_probs, model_names):
        true = np.array(true)
        probs = np.array(probs)

        # one-vs-rest
        y_true_bin = label_binarize(true, classes=np.arange(num_classes))

        # Calculate the ROC curve for each category.
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Calculate the macro-average ROC curve
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        macro_auc = auc(all_fpr, mean_tpr)

        # Plotting the macro average curve
        plt.plot(all_fpr, mean_tpr, label=f'{name} (Macro AUC={macro_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False  Positive Rate')
    plt.ylabel('True  Positive Rate')
    plt.title('ROC  Curve (Macro Average)')
    plt.legend()

    # PR
    plt.subplot(122)
    for true, probs, name in zip(true_labels, pred_probs, model_names):
        true = np.array(true)
        probs = np.array(probs)
        y_true_bin = label_binarize(true, classes=np.arange(num_classes))

        precision = dict()
        recall = dict()
        ap_score = dict()
        for i in range(num_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])
            ap_score[i] = average_precision_score(y_true_bin[:, i], probs[:, i])

        # Macro average PR curve
        mean_recall = np.linspace(0, 1, 100)
        mean_precision = np.zeros_like(mean_recall)
        for i in range(num_classes):
            mean_precision += np.interp(mean_recall, recall[i][::-1], precision[i][::-1])
        mean_precision /= num_classes
        macro_ap = auc(mean_recall, mean_precision)

        plt.plot(mean_recall, mean_precision, label=f'{name} (Macro AP={macro_ap:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall  Curve (Macro Average)')
    plt.legend()

    plt.tight_layout()
    ensure_dir(save_path)
    plt.savefig(os.path.join(save_path, 'performance_curves.png'))
    plt.close()


def plot_confusion_matrices(true_labels, pred_labels, model_names, num_classes=3, save_path='./results'):
    """Draw a multi-class confusion matrix"""
    ensure_dir(save_path)

    for true, pred, name in zip(true_labels, pred_labels, model_names):
        cm = confusion_matrix(true, pred, labels=np.arange(num_classes))

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Class {i}' for i in range(num_classes)],
                    yticklabels=[f'Class {i}' for i in range(num_classes)])
        plt.title(f'Confusion  Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'confusion_matrix_{name}.png'))
        plt.close()


# ==================== Training function ====================
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
    return total_loss / len(train_loader)

# ==================== Evaluation function ====================
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

# ==================== compute metrics ====================
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

# ==================== Single-model cross-validation training ====================

def train_and_evaluate_single_model(dataset, labels, in_feats, fp_dims, model_mode,
                                    graph_encoder_type, gat_num_heads, save_path):
    ensure_dir(save_path)
    skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)

    # Storage initialization
    best_models = []
    fold_metrics = []
    all_metrics = {'auc': [], 'f1': [], 'ba': [], 'mcc': []}

    # Modified plot data structure
    all_folds_data = {
        'true_labels': [],
        'pred_labels': [],
        'pred_probs': [],
        'test_indices': []
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
        print(f"\n=== {model_mode} + {graph_encoder_type} Fold {fold + 1}/{config.k_folds} ===")

        # Prepare data loaders
        train_loader = DataLoader(
            [dataset[i] for i in train_idx],
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.num_workers
        )
        test_loader = DataLoader(
            [dataset[i] for i in test_idx],
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.num_workers
        )

        # Initialize model
        model = MoleculeModel(
            node_feat_dim=in_feats,
            fp_dims=fp_dims,
            model_mode=model_mode,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            gnn_type=graph_encoder_type,
            gat_num_heads=gat_num_heads
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.88)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_auc = 0.0
        best_model_state = None
        current_metrics = None
        best_fold_data = None  # To store best fold's plot data

        for epoch in range(config.epochs):
            model.train()
            epoch_loss = 0.0

            # Training phase
            for graphs, batch_labels, fps in train_loader:
                if graphs is None:
                    continue

                graphs, batch_labels, node_feats, fps, edge_feats = move_batch_to_device(
                    graphs, batch_labels, fps, device)

                optimizer.zero_grad()
                outputs = model(graphs, node_feats, fps, edge_feats)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # Evaluation phase
            if (epoch + 1) % 10 == 0 or epoch == config.epochs - 1:
                model.eval()
                all_true, all_pred, all_probs = [], [], []

                with torch.no_grad():
                    for graphs, batch_labels, fps in test_loader:
                        if graphs is None:
                            continue

                        graphs, batch_labels, node_feats, fps, edge_feats = move_batch_to_device(
                            graphs, batch_labels, fps, device)

                        outputs = model(graphs, node_feats, fps, edge_feats)
                        probs = F.softmax(outputs, dim=1)
                        _, preds = torch.max(outputs, 1)

                        all_true.extend(batch_labels.cpu().numpy())
                        all_pred.extend(preds.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())

                metrics = compute_metrics(all_true, all_pred, all_probs)
                current_auc = metrics['auc']

                print(f"Epoch {epoch + 1}/{config.epochs},  Loss: {epoch_loss / len(train_loader):.4f}, "
                      f"Test AUC: {current_auc:.4f}")

                # Update best model
                if current_auc > best_auc:
                    best_auc = current_auc
                    best_model_state = deepcopy(model.state_dict())
                    current_metrics = metrics
                    best_fold_data = {
                        'true_labels': np.array(all_true),
                        'pred_labels': np.array(all_pred),
                        'pred_probs': np.array(all_probs),
                        'test_indices': test_idx
                    }

        # Save best model and data for this fold
        if best_model_state is not None:
            # Save model checkpoint
            model_path = os.path.join(save_path, f'best_model_fold_{fold + 1}.pth')
            torch.save({
                'model_state': best_model_state,
                'fold': fold + 1,
                'metrics': current_metrics,
                'config': {
                    'model_mode': model_mode,
                    'graph_encoder_type': graph_encoder_type,
                    'gat_num_heads': gat_num_heads
                },
                'plot_data': best_fold_data  # Embed plot data in model file
            }, model_path)

            # Save plot data separately
            np.savez(
                os.path.join(save_path, f'plot_data_fold_{fold + 1}.npz'),
                **best_fold_data
            )

            best_models.append(model_path)
            fold_metrics.append(current_metrics)

            # Accumulate metrics
            for key in all_metrics:
                all_metrics[key].append(current_metrics[key])

            # Store fold data for final output
            all_folds_data['true_labels'].append(best_fold_data['true_labels'])
            all_folds_data['pred_labels'].append(best_fold_data['pred_labels'])
            all_folds_data['pred_probs'].append(best_fold_data['pred_probs'])
            all_folds_data['test_indices'].append(best_fold_data['test_indices'])

            print(f"\nFold {fold + 1} Best Results:")
            print(f"AUC: {current_metrics['auc']:.4f}, F1: {current_metrics['f1']:.4f}, "
                  f"BA: {current_metrics['ba']:.4f}, MCC: {current_metrics['mcc']:.4f}")

    # Prepare final plot data (concatenated across folds)
    final_plot_data = {
        'true_labels': np.concatenate(all_folds_data['true_labels']),
        'pred_labels': np.concatenate(all_folds_data['pred_labels']),
        'pred_probs': np.vstack(all_folds_data['pred_probs']),
        'test_indices': np.concatenate(all_folds_data['test_indices'])
    }

    graph_encoder_display = graph_encoder_type.upper() if graph_encoder_type else "FP_ONLY"
    print(f"\n=== {model_mode.upper()}  + {graph_encoder_display} CV Results ===")
    print("Test Metrics (Mean ± SD):")
    print(f"AUC: {np.nanmean(all_metrics['auc']):.4f}  ± {np.nanstd(all_metrics['auc']):.4f}")
    print(f"F1: {np.mean(all_metrics['f1']):.4f}  ± {np.std(all_metrics['f1']):.4f}")
    print(f"BA: {np.mean(all_metrics['ba']):.4f}  ± {np.std(all_metrics['ba']):.4f}")
    print(f"MCC: {np.mean(all_metrics['mcc']):.4f}  ± {np.nanstd(all_metrics['mcc']):.4f}")

    return best_models, fold_metrics, all_metrics, final_plot_data


def export_metrics_to_excel(all_results, save_path):
    all_dfs = []

    for model_name, result in all_results.items():
        fold_metrics = result['fold_metrics']  # List of dicts
        df = pd.DataFrame(fold_metrics)
        df['model_name'] = model_name
        df['fold'] = range(1, len(fold_metrics) + 1)
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)

    summary_df = final_df.groupby('model_name').agg(['mean', 'std'])
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    summary_df.reset_index(inplace=True)

    with pd.ExcelWriter(save_path) as writer:
        final_df.to_excel(writer, sheet_name='Fold_Metrics', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"Model metrics have been exported to {save_path}")



def select_best_fold_lexi(fold_metrics, primary='auc', tie_breakers=('mcc', 'ba', 'f1')):

    sorted_folds = sorted(
        enumerate(fold_metrics),
        key=lambda x: x[1][primary],
        reverse=True
    )

    best_folds = [sorted_folds[0]]
    for fold in sorted_folds[1:]:
        current = best_folds[0]
        if fold[1][primary] != current[1][primary]:
            break


        is_tie = True
        for tb in tie_breakers:
            if fold[1][tb] != current[1][tb]:
                is_tie = False
                if fold[1][tb] > current[1][tb]:
                    best_folds = [fold]
                break
        if is_tie:
            best_folds.append(fold)

    return best_folds[0][0]


def main():
    # Create the necessary directories
    os.makedirs(config.visualization_path, exist_ok=True)
    os.makedirs(config.best_model_save_path, exist_ok=True)
    set_seed(config.seed)
    ensure_dir(config.best_model_save_path)

    print("Loading data...")
    df = pd.read_excel(r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\origion-data2.xlsx")
    dataset = MolecularDataset(df, add_edge_feats=config.add_edge_feats,
                               morgan_radius=config.morgan_radius,
                               morgan_nbits=config.morgan_nbits)

    labels = [dataset.labels[row['Grade']] for row in dataset.data]
    in_feats = dataset[0][0].ndata['h'].shape[1]
    fp_dims = {
        'morgan': config.morgan_nbits,
        'maccs': config.maccs_bits,
        'pharm': config.erg_length
    }

    model_configs = [
        {'model_mode': 'fusion', 'graph_encoder_type': 'gin', 'name': 'fusion_gin'},
        {'model_mode': 'fusion', 'graph_encoder_type': 'gat', 'name': 'fusion_gat'},
        {'model_mode': 'graph', 'graph_encoder_type': 'gin', 'name': 'graph_gin'},
        {'model_mode': 'graph', 'graph_encoder_type': 'gat', 'name': 'graph_gat'},
        {'model_mode': 'fp', 'graph_encoder_type': None, 'name': 'fp_only'}
    ]

    all_results = {}
    all_true_list = []
    all_pred_list = []
    all_probs_list = []
    model_names = []

    for cfg in model_configs:
        print(f"\n\n=== Training {cfg['name'].upper()} Model ===")
        save_path = os.path.join(config.best_model_save_path, cfg['name'])
        ensure_dir(save_path)

        # Training Model
        best_models, fold_metrics, all_metrics, plot_data = train_and_evaluate_single_model(
            dataset=dataset,
            labels=labels,
            in_feats=in_feats,
            fp_dims=fp_dims,
            model_mode=cfg['model_mode'],
            graph_encoder_type=cfg['graph_encoder_type'],
            gat_num_heads=config.gat_num_heads,
            save_path=save_path
        )

        # === Select the optimal fold and save the unified model ===
        best_fold_idx = select_best_fold_lexi(
            fold_metrics,
            primary='auc',
            tie_breakers=('mcc', 'ba', 'f1')
        )
        best_fold_path = best_models[best_fold_idx]


        unified_best_path = os.path.join(
            config.best_model_save_path,
            f"{cfg['name']}_best_overall.pth"
        )

        # Save the optimal model
        ckpt = torch.load(best_fold_path, map_location='cpu')
        torch.save(ckpt, unified_best_path)
        print(f"Saved unified best model to: {unified_best_path}")

        # Record Results
        all_results[cfg['name']] = {
            'best_models': best_models,
            'fold_metrics': fold_metrics,
            'all_metrics': all_metrics,
            'plot_data': plot_data,
            'best_overall_path': unified_best_path
        }

        # Collecting Visual Data
        all_true_list.append(plot_data['true_labels'])
        all_pred_list.append(plot_data['pred_labels'])
        all_probs_list.append(plot_data['pred_probs'])
        model_names.append(cfg['name'])

    print("\nAll models trained. Generating visualizations...")

    # Visualization results
    plot_combined_curves(all_true_list, all_probs_list, model_names,
                         num_classes=config.num_classes,
                         save_path=config.visualization_path)

    plot_confusion_matrices(all_true_list, all_pred_list, model_names,
                            num_classes=config.num_classes,
                            save_path=config.visualization_path)

    # Export all model metrics to excel
    export_metrics_to_excel(all_results, './GN-FG-Training_model_metrics_summary.xlsx')

    print(f"\nTraining complete! Results saved to: {config.visualization_path}")
    print(f"All best models saved to: {config.best_model_save_path}")



if __name__ == "__main__":
    config.update(
        model_mode='fusion',
        graph_encoder_type='gin',
        gat_num_heads=8,
        morgan_radius=2,
        morgan_nbits=1024,
        hidden_dim=128,
        epochs=150,
        lr=0.00015,
        dropout=0.23,
        best_model_save_path='./GN-FG-Training_best_models/',
        visualization_path='./GN-FG-Training_visualization_results/',
        batch_size=32,
        k_folds=10,
        seed=151,
        maccs_bits=167,
        erg_length=441,
        num_classes=3,
        num_workers=0
    )
    main()

