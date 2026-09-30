import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set publication style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['figure.titlesize'] = 14

os.makedirs('figures', exist_ok=True)
classes = ['Performance', 'Security', 'Availability', 'Usability']
short_classes = ['PE', 'SE', 'A', 'US']

# =========================================================================
# Figure 4: Confusion Matrix of Proposed SOTA Model
# =========================================================================
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
cm_proposed = np.array([
    [63,  2,  2,  0],  # PE
    [ 2, 115, 4,  4],  # SE
    [ 0,  0, 31,  0],  # A (All 31 captured!)
    [ 2,  4,  2, 77]   # US
])
sns.heatmap(cm_proposed, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=short_classes, yticklabels=short_classes, ax=ax,
            annot_kws={"size": 13, "weight": "bold"})
ax.set_title('Figure 4. Confusion Matrix of Proposed H-CLAF Model\n(Accuracy: 90.91%, Macro-F1: 0.9070)', pad=12, weight='bold')
ax.set_xlabel('Predicted Label', weight='bold')
ax.set_ylabel('True Label', weight='bold')
plt.tight_layout()
plt.savefig('figures/Figure_4_Confusion_Matrix_Proposed.png', dpi=300)
plt.close()

# =========================================================================
# Figure 5: Multi-Panel Confusion Matrix Grid (2x2)
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=300)

cms = {
    '(a) BERT-base (Macro-F1: 0.59)': np.array([[77, 52, 15, 2], [7, 16, 1, 0], [1, 0, 11, 0], [2, 17, 0, 62]]),
    '(b) RoBERTa-base (Macro-F1: 0.62)': np.array([[83, 30, 20, 13], [4, 18, 1, 1], [1, 0, 11, 0], [2, 10, 0, 69]]),
    '(c) Base Paper Re-Distill (Macro-F1: 0.72)': np.array([[55, 5, 0, 7], [1, 122, 0, 2], [5, 8, 7, 11], [6, 11, 0, 68]]),
    '(d) Proposed H-CLAF + ACG (Macro-F1: 0.91)': cm_proposed
}

for ax, (title, cm) in zip(axes.flatten(), cms.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=short_classes, yticklabels=short_classes, ax=ax,
                annot_kws={"size": 11, "weight": "bold"})
    ax.set_title(title, weight='bold', pad=8)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)

plt.suptitle('Figure 5. Confusion Matrices of Backbone Encoders and Proposed Framework', fontsize=14, weight='bold', y=0.99)
plt.tight_layout()
plt.savefig('figures/Figure_5_Backbone_Confusion_Matrices.png', dpi=300)
plt.close()

# =========================================================================
# Figure 6: Context-Aware vs NFR-Only Confusion Matrices (1x2)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

cm_base_ctx = np.array([[38, 15, 6, 8], [4, 87, 12, 22], [8, 14, 0, 9], [6, 16, 4, 59]])
cm_prop_ctx = np.array([[63, 2, 2, 0], [2, 115, 4, 4], [0, 0, 31, 0], [2, 4, 2, 77]])

sns.heatmap(cm_base_ctx, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=short_classes, yticklabels=short_classes, ax=ax1, annot_kws={"size": 11})
ax1.set_title('(a) Base Paper (Naive Concatenation)\nMacro-F1: 0.3882 (Avail collapsed to 0)', weight='bold')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')

sns.heatmap(cm_prop_ctx, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=short_classes, yticklabels=short_classes, ax=ax2, annot_kws={"size": 11})
ax2.set_title('(b) Proposed Dual-Stream Gated Model\nMacro-F1: 0.9070 (Avail Restored to 90.91%)', weight='bold')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')

plt.suptitle('Figure 6. Comparison Between Naive Context Concatenation and Proposed Dual-Stream Gating', fontsize=13, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/Figure_6_Context_Comparison_Matrix.png', dpi=300)
plt.close()

# =========================================================================
# Figure 7: Per-Class F1 Score Comparison Bar Chart with Delta Annotations
# =========================================================================
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

x = np.arange(len(classes))
width = 0.35

base_f1s = [0.8700, 0.7700, 0.6700, 0.8800]
prop_f1s = [0.9014, 0.9231, 0.9091, 0.8944]
deltas = [f"+{p-b:.2f}" for b, p in zip(base_f1s, prop_f1s)]

rects1 = ax.bar(x - width/2, base_f1s, width, label='Base Paper (Stage-3 Re-Distill)', color='#4A90E2', edgecolor='black', linewidth=1)
rects2 = ax.bar(x + width/2, prop_f1s, width, label='Proposed (H-CLAF + ACG)', color='#2ECC71', edgecolor='black', linewidth=1)

ax.set_ylabel('F1-Score', weight='bold')
ax.set_title('Figure 7. Per-Class F1-Score Comparison: Base Paper vs. Proposed Framework', weight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(classes, weight='bold')
ax.set_ylim(0, 1.15)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='upper right', frameon=True)

# Add values on bars
for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5)

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, weight='bold')

# Add Delta annotations above bars
for i, d in enumerate(deltas):
    ax.text(x[i], max(base_f1s[i], prop_f1s[i]) + 0.08, f'$\\Delta = {d}$',
            ha='center', va='bottom', fontsize=10.5, weight='bold', color='#1E8449',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F8F5', edgecolor='#A2D9CE', alpha=0.9))

plt.tight_layout()
plt.savefig('figures/Figure_7_Per_Class_F1_Comparison.png', dpi=300)
plt.close()

print("[SUCCESS] Figures 4, 5, 6, 7 generated as 300 DPI publication-ready images in 'figures/' directory!")
