import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GINConv, AvgPooling, GATConv
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from torch.utils.data import Dataset
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

# ==================== Configuration Parameter System ====================
class Config:


    def __init__(self):

        self.seed = 151  
        self.model_mode = 'graph'  #  'fusion'/'graph'/'fp'
        self.num_classes = 3
        self.k_folds = 10
        self.epochs = 150

        self.batch_size = 32
        self.num_workers = 0
        self.add_edge_feats = True

        self.morgan_radius = 2
        self.morgan_nbits = 1024
        self.maccs_bits = 167
        self.erg_length = 441

        self.hidden_dim = 128
        self.dropout = 0.23
        self.gnn_num_layers = 4

        self.lr = 0.00015
        self.best_model_save_path = './best_models/'
        self.visualization_path = './visualization_results/'

        self.graph_encoder_type = 'gat'  # 'gin', 'gat'
        self.gat_num_heads = 8

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
        self.atom_featurizer = CanonicalAtomFeaturizer()  # Atomic Feature Extractor
        self.add_edge_feats = add_edge_feats
        if self.add_edge_feats:
            self.bond_featurizer = CanonicalBondFeaturizer()  # Key Feature Extractor
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

        # Calculate fingerprint features
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
            self.use_residual.append(i > 0)  # Use residual connections starting from the second layer.
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
        self.pool = AvgPooling()  # Level pooling layer

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
        self.residual_projections = nn.ModuleList()  # Used for projection residual connection

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
        # Choose the appropriate graph encoder based on gnn_type
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
