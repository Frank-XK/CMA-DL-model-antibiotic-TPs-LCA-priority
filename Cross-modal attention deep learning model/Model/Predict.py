#fusion_gin Predicttion

import os
import torch
import torch.nn.functional as F
import pandas as pd
from rdkit import Chem
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from model_utils import MoleculeModel, mol_to_fingerprints

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_map = {0: 'Ⅰ', 1: 'Ⅱ', 2: 'Ⅲ'}

# Five model configurations
MODEL_CONFIGS = {
    'fusion_gin': {'model_mode': 'fusion', 'gnn_type': 'gin', 'gat_num_heads': 8},
    'fusion_gat': {'model_mode': 'fusion', 'gnn_type': 'gat', 'gat_num_heads': 8},
    'graph_gin': {'model_mode': 'graph', 'gnn_type': 'gin', 'gat_num_heads': 8},
    'graph_gat': {'model_mode': 'graph', 'gnn_type': 'gat', 'gat_num_heads': 8},
    'fp_only': {'model_mode': 'fp', 'gnn_type': None, 'gat_num_heads': 0},
}

def load_model(model_path, node_feat_dim, fp_dims, model_mode, gnn_type, gat_num_heads):
    checkpoint = torch.load(model_path, map_location=device)
    hidden_dim = checkpoint['model_state']['classifier.0.weight'].shape[1]
    model = MoleculeModel(
        node_feat_dim=node_feat_dim,
        fp_dims=fp_dims,
        model_mode=model_mode,
        hidden_dim=hidden_dim,
        num_classes=3,
        gnn_type=gnn_type if gnn_type else 'gin',  # fp_only
        gat_num_heads=gat_num_heads
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def preprocess_smiles(smiles, add_edge_feats=True, morgan_radius=2, morgan_nbits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer() if add_edge_feats else None
    if add_edge_feats:
        g = smiles_to_bigraph(smiles, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
    else:
        g = smiles_to_bigraph(smiles, node_featurizer=atom_featurizer)
    fps = mol_to_fingerprints(mol, morgan_radius, morgan_nbits)
    return g, fps

def predict(model, g, fps, model_mode):
    if model_mode != 'fp':
        g = g.to(device)
        node_feats = g.ndata['h'].to(device)
    else:
        node_feats = None
    fps = {k: v.unsqueeze(0).to(device) for k, v in fps.items()}
    with torch.no_grad():
        if model_mode == 'fusion':
            output = model(g, node_feats, fps)
        elif model_mode == 'graph':
            output = model(g, node_feats, None)
        elif model_mode == 'fp':
            output = model(None, None, fps)
        else:
            raise ValueError("Invalid model_mode")
        probs = F.softmax(output, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
    return pred_label, probs.cpu().numpy()

def main():
    # ====== Path and Model ======
    model_name = 'fusion_gin'  # fusion_gin, fusion_gat, graph_gin, graph_gat, fp_only
    model_path =  fr"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\best_models-AMR\{model_name}_best_overall-AMR.pth"
    input_csv = r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\Biotransform\abioticbio_transformation_high_similarity_results-0.7.csv"
    output_csv = r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\Biotransform\abioticbio_ransformation_predictions_fusion_gin_AMR.csv"

    config = MODEL_CONFIGS[model_name]
    model_mode = config['model_mode']
    gnn_type = config['gnn_type']
    gat_num_heads = config['gat_num_heads']
    add_edge_feats = model_mode in ['fusion', 'graph']

    fp_dims = {
        'morgan': 1024,
        'maccs': 167,
        'pharm': 441
    }

    df = pd.read_csv(input_csv)
    smiles_list = df['SMILES'].tolist()

    # Obtain node feature dimensions (fp_only mode does not require a graph)
    if model_mode != 'fp':
        atom_featurizer = CanonicalAtomFeaturizer()
        first_graph = smiles_to_bigraph(smiles_list[0], node_featurizer=atom_featurizer)
        node_feat_dim = first_graph.ndata['h'].shape[1]
    else:
        node_feat_dim = 128  # fp_only mode can be set arbitrarily

    model = load_model(model_path, node_feat_dim, fp_dims, model_mode, gnn_type, gat_num_heads)

    preds = []
    for smi in smiles_list:
        g, fps = preprocess_smiles(smi, add_edge_feats)
        if g is None and model_mode != 'fp':
            preds.append('Invalid_SMILES')
            continue
        pred_label, prob = predict(model, g, fps, model_mode)
        preds.append(label_map[pred_label])

    df['Predicted_Grade'] = preds
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Prediction complete, results saved to:{output_csv}")
    print(preds[:10])
if __name__ == '__main__':
    main()



#Submodel Prediction Predicttion


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from rdkit import Chem
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgl.nn import GINConv, AvgPooling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tag mapping
label_map = {0: 'Ⅰ', 1: 'Ⅱ'}

# ---------------- Model definition ----------------

class FingerprintEncoder(nn.Module):
    def __init__(self, fp_dims, emb_dim=128, dropout=0.23):
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
    def __init__(self, in_feats, hidden_dim=128, num_layers=4, dropout=0.23):
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
    def __init__(self, node_feat_dim, fp_dims, hidden_dim=128, dropout=0.23):
        super().__init__()
        self.graph_encoder = GIN_Encoder(node_feat_dim, hidden_dim, num_layers=4, dropout=dropout)
        self.fp_encoder = FingerprintEncoder(fp_dims, hidden_dim, dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=2, dropout=dropout, batch_first=False)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
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
    def __init__(self, node_feat_dim, fp_dims, hidden_dim=128, num_classes=2, dropout=0.23):
        super().__init__()
        self.encoder = CrossModalFusionEncoder(node_feat_dim, fp_dims, hidden_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, g, node_feats, fps, edge_feats=None):
        x = self.encoder(g, node_feats, fps)
        return self.classifier(x)

# ---------------- fingerprint computing ----------------

from rdkit.Chem import AllChem, MACCSkeys

def mol_to_fingerprints(mol, morgan_radius=2, morgan_nbits=1024):
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=morgan_radius, nBits=morgan_nbits)
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    pharm_fp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
    return {
        'morgan': torch.tensor(morgan_fp, dtype=torch.float32),
        'maccs': torch.tensor(maccs_fp, dtype=torch.float32),
        'pharm': torch.tensor(pharm_fp, dtype=torch.float32)
    }

# ---------------- Prediction Related ----------------

def load_model(model_path, node_feat_dim, fp_dims, hidden_dim, num_classes=2):
    checkpoint = torch.load(model_path, map_location=device)
    model = MoleculeModel(node_feat_dim, fp_dims, hidden_dim, num_classes).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def preprocess_smiles(smiles, add_edge_feats=True, morgan_radius=2, morgan_nbits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    atom_featurizer = CanonicalAtomFeaturizer()
    bond_featurizer = CanonicalBondFeaturizer() if add_edge_feats else None
    if add_edge_feats:
        g = smiles_to_bigraph(smiles, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
    else:
        g = smiles_to_bigraph(smiles, node_featurizer=atom_featurizer)
    fps = mol_to_fingerprints(mol, morgan_radius, morgan_nbits)
    return g, fps

def predict(model, g, fps):
    g = g.to(device)
    node_feats = g.ndata['h'].to(device)
    fps = {k: v.unsqueeze(0).to(device) for k, v in fps.items()}
    with torch.no_grad():
        output = model(g, node_feats, fps)
        probs = F.softmax(output, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
    return pred_label, probs.cpu().numpy()

# ---------------- main function ----------------

def main():
    # Modify path
    model_path = r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\best_models-HuhR\fusion_gin_best_overall-HuhR.pth"
    input_csv = r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\origion-data2.csv"
    output_csv = r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\Biotransform\predictions-Origion-HuhR.csv"

    fp_dims = {
        'morgan': 1024,
        'maccs': 167,
        'pharm': 441
    }

    df = pd.read_csv(input_csv)
    smiles_list = df['SMILES'].tolist()

    # Obtain node feature dimensions
    atom_featurizer = CanonicalAtomFeaturizer()
    node_feat_dim = None
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            g = smiles_to_bigraph(smi, node_featurizer=atom_featurizer)
            node_feat_dim = g.ndata['h'].shape[1]
            break
    if node_feat_dim is None:
        raise ValueError("The input data does not contain valid SMILES, so the node feature dimensions cannot be obtained")

    # Read the hidden_dim parameter from the model
    checkpoint = torch.load(model_path, map_location=device)
    hidden_dim = checkpoint['model_state']['classifier.0.weight'].shape[1]

    model = load_model(model_path, node_feat_dim, fp_dims, hidden_dim)

    preds = []
    for smi in smiles_list:
        g, fps = preprocess_smiles(smi, add_edge_feats=True)
        if g is None or fps is None:
            preds.append('Invalid_SMILES')
            continue
        pred_label, prob = predict(model, g, fps)
        preds.append(label_map[pred_label])

    df['Predicted_Grade'] = preds
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Prediction complete, results saved to:{output_csv}")
    print("Examples of the first 10 prediction results:", preds[:10])

if __name__ == '__main__':
    main()

