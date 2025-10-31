import os
import numpy as np
import pandas as pd
import torch
import dgl
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

from dgl.nn.pytorch.explain import GNNExplainer
from rdkit.Chem import GetPeriodicTable
from rdkit import Chem
from rdkit.Chem import Draw

from GN_FG_Training import Config, MoleculeModel, MolecularDataset, set_seed

# ================= Parameter settings =================
DATA_PATH = r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\origion-data2.xlsx"
MODEL_PATH = r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\GN-FG-Training_best_models\graph_gin_best_overall.pth"
RESULTS_DIR = r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\explain_results_Fig"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ================= Public utility functions =================
def _bond_type_to_str(x: int) -> str:
    m = {1:"single", 2:"double", 3:"triple", 4:"aromatic"}
    try:
        return m.get(int(x), str(int(x)))
    except Exception:
        return str(x)

def _normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    vmin, vmax = np.min(x), np.max(x)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - vmin) / (vmax - vmin + 1e-12)

def _atom_num_to_symbol(atomic_num: int) -> str:
    ptable = GetPeriodicTable()
    try:
        return ptable.GetElementSymbol(int(atomic_num))
    except Exception:
        return str(atomic_num)

def _get_node_labels(g_cpu: dgl.DGLGraph) -> dict:
    if "atomic_num" in g_cpu.ndata:
        try:
            return {i: _atom_num_to_symbol(int(g_cpu.ndata["atomic_num"][i].item()))
                    for i in range(g_cpu.num_nodes())}
        except Exception:
            pass
    return {i: str(i) for i in range(g_cpu.num_nodes())}

def _get_edge_labels(g_cpu: dgl.DGLGraph, src: np.ndarray, dst: np.ndarray) -> dict:
    edge_labels = {}
    has_aromatic = 'is_aromatic' in g_cpu.edata
    key_candidates = ["bond_type", "bond_order", "order", "type"]
    found_key = None
    for k in key_candidates:
        if k in g_cpu.edata:
            found_key = k
            break
    for eid, (u, v) in enumerate(zip(src, dst)):
        parts = []
        if found_key is not None:
            try:
                val = g_cpu.edata[found_key][eid].item()
                parts.append(_bond_type_to_str(int(val)))
            except Exception:
                pass
        if has_aromatic:
            try:
                arom = bool(g_cpu.edata['is_aromatic'][eid].item())
                if arom:
                    parts.append("aromatic")
            except Exception:
                pass
        edge_labels[(int(u), int(v))] = "; ".join([p for p in parts if p]) if parts else ""
    return edge_labels

def _build_norm(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        return mpl.colors.Normalize(vmin=0.0, vmax=1.0), None
    vmin, vmax = float(values.min()), float(values.max())
    if abs(vmax - vmin) < 1e-12:
        eps = 1e-12
        return mpl.colors.Normalize(vmin=vmin - eps, vmax=vmax + eps), [vmin]
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax), None

def _draw_explained_graph(nx_g, pos,
                          node_imp_raw, edge_imp_raw, edge_imp_norm,
                          node_labels, edge_labels, save_path,
                          node_norm=None, edge_norm=None,
                          draw_colorbar=False):
    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.weight"] = "bold"

    reds = plt.cm.Reds
    blues = plt.cm.Blues

    if node_norm is None:
        node_norm, node_ticks = _build_norm(node_imp_raw)
    else:
        node_ticks = None
    if edge_norm is None:
        edge_norm, edge_ticks = _build_norm(edge_imp_raw)
    else:
        edge_ticks = None

    node_colors = reds(node_norm(node_imp_raw))
    edge_colors = blues(edge_norm(edge_imp_raw))

    nx.draw_networkx_nodes(
        nx_g, pos,
        node_color=node_colors,
        node_size=800,
        linewidths=1.2,
        edgecolors="#333333",
        ax=ax
    )

    widths = 1.5 + 3.5 * edge_imp_norm
    nx.draw_networkx_edges(
        nx_g, pos,
        edge_color=edge_colors,
        width=widths,
        arrows=False,
        ax=ax
    )

    nx.draw_networkx_labels(
        nx_g, pos,
        labels=node_labels,
        font_size=20,
        font_color="black",
        font_family="Times New Roman"
    )

    nx.draw_networkx_edge_labels(
        nx_g, pos,
        edge_labels=edge_labels,
        font_size=20,
        font_color="black",
        font_family="Times New Roman"
    )

    if draw_colorbar:
        sm_nodes = mpl.cm.ScalarMappable(cmap=reds, norm=node_norm)
        sm_edges = mpl.cm.ScalarMappable(cmap=blues, norm=edge_norm)

        cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.035, pad=0.2)
        cbar_nodes.set_label("Node Importance", fontsize=25, fontweight='bold', family='Times New Roman')
        if node_ticks is not None:
            cbar_nodes.set_ticks(node_ticks)
        cbar_nodes.ax.tick_params(labelsize=20)

        cbar_edges = plt.colorbar(sm_edges, ax=ax, fraction=0.035, pad=0.2)
        cbar_edges.set_label("Edge Importance", fontsize=25, fontweight='bold', family='Times New Roman')
        if edge_ticks is not None:
            cbar_edges.set_ticks(edge_ticks)
        cbar_edges.ax.tick_params(labelsize=20)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def draw_unified_colorbars(node_norm, edge_norm, save_path):
    fig, ax = plt.subplots(figsize=(4, 6))

    reds = plt.cm.Reds
    blues = plt.cm.Blues

    sm_nodes = mpl.cm.ScalarMappable(cmap=reds, norm=node_norm)
    sm_edges = mpl.cm.ScalarMappable(cmap=blues, norm=edge_norm)

    cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.4, pad=0.2)
    cbar_nodes.set_label("Node Importance", fontsize=25, fontweight='bold', family='Times New Roman')
    cbar_nodes.ax.tick_params(labelsize=25)

    cbar_edges = plt.colorbar(sm_edges, ax=ax, fraction=0.4, pad=0.2)
    cbar_edges.set_label("Edge Importance", fontsize=25, fontweight='bold', family='Times New Roman')
    cbar_edges.ax.tick_params(labelsize=25)

    ax.remove()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"The unified legend has been saved to {save_path}")

def highlight_with_rdkit(smiles, node_imp_raw, edge_imp_raw, src, dst, topk=5, save_path="highlighted.png"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES Parsing failed: {smiles}")

    n_atoms = mol.GetNumAtoms()
    topk_nodes = min(max(1, int(topk)), max(1, len(node_imp_raw)))
    topk_edges = min(max(1, int(topk)), max(1, len(edge_imp_raw)))

    node_order = np.argsort(node_imp_raw)
    top_nodes = node_order[-topk_nodes:]
    highlight_atoms = [int(x) for x in sorted(top_nodes.tolist()) if int(x) < n_atoms]

    edge_pairs = []
    for i in range(len(src)):
        u, v = int(src[i]), int(dst[i])
        if u == v:
            continue
        a, b = (u, v) if u <= v else (v, u)
        edge_pairs.append((a, b, i))
    pair_to_best = {}
    for a, b, idx in edge_pairs:
        key = (int(a), int(b))
        val = edge_imp_raw[idx] if idx < len(edge_imp_raw) else 0.0
        if key not in pair_to_best or val > pair_to_best[key][0]:
            pair_to_best[key] = (val, idx)
    undirected_edges = sorted([(k[0], k[1], v[1], v[0]) for k, v in pair_to_best.items()])

    highlight_bonds = []
    used_pairs = set()
    for u, v, orig_idx, imp in undirected_edges:
        if (u in highlight_atoms) and (v in highlight_atoms):
            b = mol.GetBondBetweenAtoms(int(u), int(v))
            if b is not None:
                bid = int(b.GetIdx())
                highlight_bonds.append(bid)
                used_pairs.add((u, v))
    if len(highlight_bonds) < topk_edges:
        sorted_edges = sorted(undirected_edges, key=lambda x: x[3], reverse=True)
        for u, v, orig_idx, imp in sorted_edges:
            if len(highlight_bonds) >= topk_edges:
                break
            if (u, v) in used_pairs:
                continue
            b = mol.GetBondBetweenAtoms(int(u), int(v))
            if b is None:
                continue
            bid = int(b.GetIdx())
            highlight_bonds.append(bid)
            used_pairs.add((u, v))

    highlight_atoms = sorted(list(dict.fromkeys([int(x) for x in highlight_atoms])))
    highlight_bonds = sorted(list(dict.fromkeys([int(x) for x in highlight_bonds])))

    img = Draw.MolToImage(
        mol,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
        size=(400, 400)
    )
    img.save(save_path)
    print(f"[RDKit] Save the highlight image to {save_path} (atoms={len(highlight_atoms)}, bonds={len(highlight_bonds)})")

def explain_graph_on_sample(sample_idx=0, num_hops=2, add_self_loop=True,
                            results_dir=RESULTS_DIR, prefix="gin",
                            node_norm=None, edge_norm=None,
                            draw_colorbar=True):
    set_seed(151)
    df = pd.read_excel(DATA_PATH)
    dataset = MolecularDataset(df, add_edge_feats=True,
                               morgan_radius=2,
                               morgan_nbits=1024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_feats = dataset[0][0].ndata['h'].shape[1]
    fp_dims = {'morgan': 1024, 'maccs': 167, 'pharm': 441}

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = MoleculeModel(node_feat_dim=in_feats,
                          fp_dims=fp_dims,
                          model_mode='graph',
                          hidden_dim=128,
                          num_classes=3,
                          gnn_type='gin').to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    class WrapperGINEncoder(torch.nn.Module):
        def __init__(self, gin_encoder):
            super().__init__()
            self.gin_encoder = gin_encoder
        def forward(self, graph, feat, **kwargs):
            return self.gin_encoder(graph, feat)

    wrapped_encoder = WrapperGINEncoder(model.graph_encoder)
    explainer = GNNExplainer(wrapped_encoder, num_hops=num_hops)

    g, label, fps = dataset[sample_idx]
    g = g.to(device)
    node_feats = g.ndata['h'].to(device)

    g_exp = dgl.add_self_loop(g) if add_self_loop else g

    feat_mask, edge_mask = explainer.explain_graph(g_exp, node_feats)

    edge_imp_raw = edge_mask.detach().cpu().numpy().astype(float)
    edge_imp_norm = _normalize01(edge_imp_raw)

    g_cpu = g_exp.to('cpu')
    nx_g = g_cpu.to_networkx()
    pos = nx.spring_layout(nx_g, seed=151)

    src_e, dst_e = g_cpu.edges()
    src = src_e.numpy() if isinstance(src_e, torch.Tensor) else np.asarray(src_e)
    dst = dst_e.numpy() if isinstance(dst_e, torch.Tensor) else np.asarray(dst_e)

    node_imp_raw = np.zeros(g_cpu.num_nodes(), dtype=np.float64)
    for i, w in enumerate(edge_imp_raw):
        u, v = int(src[i]), int(dst[i])
        node_imp_raw[u] += float(w)
        node_imp_raw[v] += float(w)

    node_labels = _get_node_labels(g_cpu)
    edge_labels = _get_edge_labels(g_cpu, src, dst)

    save_path = os.path.join(results_dir, f"{prefix}_graph_explained_sample{sample_idx}.png")
    _draw_explained_graph(
        nx_g, pos,
        node_imp_raw=node_imp_raw,
        edge_imp_raw=edge_imp_raw,
        edge_imp_norm=edge_imp_norm,
        node_labels=node_labels,
        edge_labels=edge_labels,
        save_path=save_path,
        node_norm=node_norm,
        edge_norm=edge_norm,
        draw_colorbar=draw_colorbar
    )
    print(f"Explanation diagram has been saved to {save_path}")

    smiles_col = None
    for c in df.columns:
        if 'smile' in c.lower():
            smiles_col = c
            break
    if smiles_col is not None:
        smiles = df.iloc[sample_idx][smiles_col]
        highlight_path = os.path.join(results_dir, f"{prefix}_rdkit_highlight_sample{sample_idx}.png")
        highlight_with_rdkit(smiles, node_imp_raw, edge_imp_raw, src, dst, topk=5, save_path=highlight_path)

def collect_global_importance_ranges(sample_indices, dataset, model, device, num_hops=2, add_self_loop=True):
    all_node_imps = []
    all_edge_imps = []

    class WrapperGINEncoder(torch.nn.Module):
        def __init__(self, gin_encoder):
            super().__init__()
            self.gin_encoder = gin_encoder
        def forward(self, graph, feat, **kwargs):
            return self.gin_encoder(graph, feat)

    wrapped_encoder = WrapperGINEncoder(model.graph_encoder)
    explainer = GNNExplainer(wrapped_encoder, num_hops=num_hops)

    for sample_idx in sample_indices:
        g, label, fps = dataset[sample_idx]
        g = g.to(device)
        node_feats = g.ndata['h'].to(device)
        g_exp = dgl.add_self_loop(g) if add_self_loop else g

        feat_mask, edge_mask = explainer.explain_graph(g_exp, node_feats)
        edge_imp_raw = edge_mask.detach().cpu().numpy().astype(float)
        all_edge_imps.append(edge_imp_raw)

        g_cpu = g_exp.to('cpu')
        src_e, dst_e = g_cpu.edges()
        src = src_e.numpy() if isinstance(src_e, torch.Tensor) else np.asarray(src_e)
        dst = dst_e.numpy() if isinstance(dst_e, torch.Tensor) else np.asarray(dst_e)

        node_imp_raw = np.zeros(g_cpu.num_nodes(), dtype=np.float64)
        for i, w in enumerate(edge_imp_raw):
            u, v = int(src[i]), int(dst[i])
            node_imp_raw[u] += float(w)
            node_imp_raw[v] += float(w)
        all_node_imps.append(node_imp_raw)

    all_node_imps_concat = np.concatenate(all_node_imps)
    all_edge_imps_concat = np.concatenate(all_edge_imps)

    node_norm, _ = _build_norm(all_node_imps_concat)
    edge_norm, _ = _build_norm(all_edge_imps_concat)

    return node_norm, edge_norm

def main():
    set_seed(151)
    df = pd.read_excel(DATA_PATH)
    dataset = MolecularDataset(df, add_edge_feats=True,
                               morgan_radius=2,
                               morgan_nbits=1024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_feats = dataset[0][0].ndata['h'].shape[1]
    fp_dims = {'morgan': 1024, 'maccs': 167, 'pharm': 441}

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = MoleculeModel(node_feat_dim=in_feats,
                          fp_dims=fp_dims,
                          model_mode='graph',
                          hidden_dim=128,
                          num_classes=3,
                          gnn_type='gin').to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    sample_indices = [35,59,32,50,9,46,48,53,0,4,12,26]

    print("Begin calculating the global normalized range for all samples....")
    node_norm, edge_norm = collect_global_importance_ranges(sample_indices, dataset, model, device)

    print("Begin drawing explanatory diagrams one by one (excluding legends)....")
    for idx in sample_indices:
        print(f"Explanation Sample {idx} ...")
        explain_graph_on_sample(sample_idx=idx, num_hops=2, add_self_loop=True,
                                results_dir=RESULTS_DIR, prefix="gin",
                                node_norm=node_norm, edge_norm=edge_norm,
                                draw_colorbar=False)

    print("Draw a uniform color bar legend...")
    legend_path = os.path.join(RESULTS_DIR, "unified_colorbar_legend.png")
    draw_unified_colorbars(node_norm, edge_norm, legend_path)

if __name__ == "__main__":
    main()
