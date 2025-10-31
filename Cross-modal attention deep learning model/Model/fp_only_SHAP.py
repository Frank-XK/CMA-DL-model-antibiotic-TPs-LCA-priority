import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from GN_FG_Training import MoleculeModel, MolecularDataset, config, set_seed

# =============== Global font configuration ===============
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14

def set_bold_font(ax):
    ax.title.set_fontweight('bold')
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

# ==================== path ====================
data_path = r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\origion-data2.xlsx"
model_path = r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\GN-FG-Training_best_models\fp_only_best_overall.pth"
output_dir = r"C:\Users\Qikun Pu\PyCharmMiscProject\Qikun\explain_results_FG_gradient_fixed_v2"

os.makedirs(output_dir, exist_ok=True)

# ==================== Loading data ====================
set_seed(config.seed)
df = pd.read_excel(data_path)
dataset = MolecularDataset(df, add_edge_feats=config.add_edge_feats,
                           morgan_radius=config.morgan_radius,
                           morgan_nbits=config.morgan_nbits)

in_feats = dataset[0][0].ndata['h'].shape[1]

fp_dims = {
    'morgan': config.morgan_nbits,
    'maccs': config.maccs_bits,
    'pharm': config.erg_length
}

feature_names = []
feature_names += [f"Morgan_{i}" for i in range(fp_dims['morgan'])]
feature_names += [f"MACCS_{i}" for i in range(fp_dims['maccs'])]
feature_names += [f"ErG_{i}" for i in range(fp_dims['pharm'])]

# ==================== Loading Model ====================
checkpoint = torch.load(model_path, map_location='cpu')
model = MoleculeModel(node_feat_dim=in_feats,
                      fp_dims=fp_dims,
                      model_mode='fp',
                      hidden_dim=config.hidden_dim,
                      num_classes=config.num_classes).to('cpu')
model.load_state_dict(checkpoint['model_state'])
model.eval()

# ==================== Packaging Model ====================
class WrappedModel(nn.Module):
    def __init__(self, original_model, fp_dims):
        super().__init__()
        self.model = original_model
        self.fp_dims = fp_dims

    def forward(self, x):
        morgan_dim = self.fp_dims['morgan']
        maccs_dim = self.fp_dims['maccs']
        pharm_dim = self.fp_dims['pharm']

        fps = {
            'morgan': x[:, :morgan_dim],
            'maccs': x[:, morgan_dim:morgan_dim+maccs_dim],
            'pharm': x[:, morgan_dim+maccs_dim:]
        }
        return self.model(None, None, fps)

wrapped_model = WrappedModel(model, fp_dims)
wrapped_model.eval()

# ==================== Auxiliary functions ====================
def get_fp_tensor(idx):
    _, label, fps = dataset[idx]
    concat_fp = torch.cat([fps['morgan'], fps['maccs'], fps['pharm']])
    return concat_fp.unsqueeze(0).float(), int(label)

# ==================== Prepare background data ====================
background = []
for i in range(min(141, len(dataset))):
    x, _ = get_fp_tensor(i)
    background.append(x.squeeze(0).numpy())
background = np.array(background)
background_tensor = torch.tensor(background, dtype=torch.float32)

# ==================== GradientExplainer ====================
explainer = shap.GradientExplainer(wrapped_model, background_tensor)

# ==================== Calculate SHAP value ====================
all_importances = []
all_labels = []
all_preds = []

for idx in range(len(dataset)):
    x, label = get_fp_tensor(idx)
    shap_values = explainer.shap_values(x)  

    pred_probs = torch.softmax(wrapped_model(x), dim=1).detach().cpu().numpy()
    pred_class = np.argmax(pred_probs)

    if isinstance(shap_values, list):
        class_shap = shap_values[pred_class][0]
    else:
        if shap_values.ndim == 3:
            # (1, feature_dim, num_classes)
            class_shap = shap_values[0, :, pred_class]
        elif shap_values.ndim == 2:
            # (1, feature_dim)
            class_shap = shap_values[0]
        else:
            raise ValueError(f"Unexpected shap_values shape: {shap_values.shape}")

    all_importances.append(class_shap)
    all_labels.append(label)
    all_preds.append(pred_class)

all_importances = np.array(all_importances)  # shape (num_samples, feature_dim)
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# ==================== Excel ====================
df_importance = pd.DataFrame(all_importances, columns=feature_names)
df_importance["label"] = all_labels
df_importance["pred"] = all_preds
excel_path = os.path.join(output_dir, "feature_importance_GradientExplainer_named_fixed_v2.xlsx")
df_importance.to_excel(excel_path, index=False)
print(f"GradientExplainer Feature importance to {excel_path}")

# ==================== Beeswarm Plot ====================
max_display_features = 15

for cls in np.unique(all_preds):
    mask = all_preds == cls
    plt.figure(figsize=(4, 7))
    shap.summary_plot(all_importances[mask], features=all_importances[mask],
                      feature_names=feature_names, show=False,
                      max_display=max_display_features)
    plt.title(f"Beeswarm Plot - Class {cls}", fontsize=25, fontweight='bold')
    plt.xlabel("SHAP value (GradientExplainer)", fontsize=25, fontweight='bold')
    plt.ylabel("Features", fontsize=25, fontweight='bold')
    ax = plt.gca()
    set_bold_font(ax)
    plt.savefig(os.path.join(output_dir, f"beeswarm_class_{cls}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
print(f"Beeswarm to {output_dir}")

# ==================== Bar Plot ====================
def plot_top20_bar(importance_array, feature_names, class_label, save_dir):
    mean_abs_imp = np.mean(np.abs(importance_array), axis=0)
    top20_idx = np.argsort(-mean_abs_imp)[:15]
    top20_features = [feature_names[i] for i in top20_idx]
    top20_values = mean_abs_imp[top20_idx]

    plt.figure(figsize=(8, 5))
    bars = plt.barh(range(len(top20_values)), top20_values[::-1], color='skyblue')
    plt.yticks(range(len(top20_values)), top20_features[::-1], fontsize=12, fontweight='bold')
    plt.xlabel("Mean |SHAP value|", fontsize=14, fontweight='bold')
    plt.title(f"Top 20 Important Features - Class {class_label}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    ax = plt.gca()
    set_bold_font(ax)
    save_path = os.path.join(save_dir, f"barplot_top20_class_{class_label}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 已保存 Bar Plot 到 {save_path}")

for cls in np.unique(all_preds):
    mask = all_preds == cls
    plot_top20_bar(all_importances[mask], feature_names, cls, output_dir)

# ==================== Force Plot ====================
sample_idx = 100
topN = 10

x, label = get_fp_tensor(sample_idx)
logits = torch.softmax(wrapped_model(x), dim=1).detach().cpu().numpy()
num_classes = logits.shape[1]

for cls in range(num_classes):
    shap_vals = explainer.shap_values(x)
    if isinstance(shap_vals, list):
        if len(shap_vals) > cls:
            shap_vals_cls = shap_vals[cls][0]
        else:
            shap_vals_cls = shap_vals[0]
    else:
        if shap_vals.ndim == 3:
            shap_vals_cls = shap_vals[0, :, cls]
        else:
            shap_vals_cls = shap_vals[0]

    topN_idx = np.argsort(-np.abs(shap_vals_cls))[:topN]
    topN_shap_values = shap_vals_cls[topN_idx]
    topN_features = [feature_names[i] for i in topN_idx]

    force_html_path = os.path.join(output_dir, f"force_plot_sample{sample_idx}_class{cls}_top{topN}.html")
    shap.save_html(force_html_path,
                   shap.force_plot(base_value=0,
                                   shap_values=topN_shap_values,
                                   features=topN_features,
                                   feature_names=topN_features))
    print(f"force plot HTML to {force_html_path}")

    plt.figure(figsize=(12, 3))
    shap.force_plot(base_value=0,
                    shap_values=topN_shap_values,
                    features=topN_features,
                    feature_names=topN_features,
                    matplotlib=True, show=False)
    ax = plt.gca()
    set_bold_font(ax)
    png_path = os.path.join(output_dir, f"force_plot_sample{sample_idx}_class{cls}_top{topN}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" force plot PNG to {png_path}")
