import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

fig = plt.figure(figsize=(15, 6.2), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 15)
ax.set_ylim(0, 6.2)
ax.axis("off")

# Global background styling
plt.rcParams["font.family"] = "DejaVu Sans"

# ==============================================================================
# LEFT PANEL: GENERALIZED FORMULATION WORKFLOW
# ==============================================================================
# Outer container Left
left_box = FancyBboxPatch((0.2, 0.2), 10.3, 5.8, boxstyle="round,pad=0.15,rounding_size=0.2",
                          edgecolor="#2c3e50", facecolor="#f8f9fa", linewidth=1.5)
ax.add_patch(left_box)

# Left Header
ax.text(5.35, 5.75, "Formal Dataset Traceability & Pair Formulation Framework", 
        ha="center", va="center", fontsize=13, fontweight="bold", color="#1a252f")

# --- Step 1: Raw Functional Specification (FR Anchor) ---
fr_box = FancyBboxPatch((0.5, 4.4), 9.7, 1.0, boxstyle="round,pad=0.1,rounding_size=0.15",
                        edgecolor="#27ae60", facecolor="#eafaf1", linewidth=1.2)
ax.add_patch(fr_box)
# Tag
fr_tag = FancyBboxPatch((0.7, 4.95), 2.2, 0.32, boxstyle="round,pad=0.05,rounding_size=0.08",
                        edgecolor="#27ae60", facecolor="#27ae60")
ax.add_patch(fr_tag)
ax.text(1.8, 5.11, "FUNCTIONAL CONTEXT (FR)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")

ax.text(0.8, 4.65, r"$\mathbf{c}_i^{\mathrm{FR}}$ :  $\left[\mathrm{CLS}\right], t_1^{\mathrm{FR}}, t_2^{\mathrm{FR}}, \dots, t_N^{\mathrm{FR}}, \left[\mathrm{SEP}\right]$", 
        fontsize=11, color="#1e3a1e")
ax.text(0.8, 4.45, "Example: \"The distributed telemetry engine shall log authenticated flight sensor metrics.\"", 
        fontsize=9, fontstyle="italic", color="#2c3e50")

# Arrow Down 1
ax.annotate("", xy=(5.35, 3.85), xytext=(5.35, 4.35),
            arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#2c3e50", mutation_scale=15))
ax.text(5.5, 4.1, "Traceability Link Resolution Across Systems (P1-P15)", fontsize=8.5, color="#555555", fontstyle="italic")

# --- Step 2: Associated Quality Constraints (NFR Stream Candidates) ---
nfr_box = FancyBboxPatch((0.5, 1.95), 9.7, 1.85, boxstyle="round,pad=0.1,rounding_size=0.15",
                         edgecolor="#2980b9", facecolor="#ebf5fb", linewidth=1.2)
ax.add_patch(nfr_box)

nfr_tag = FancyBboxPatch((0.7, 3.52), 3.0, 0.32, boxstyle="round,pad=0.05,rounding_size=0.08",
                         edgecolor="#2980b9", facecolor="#2980b9")
ax.add_patch(nfr_tag)
ax.text(2.2, 3.68, "LINKED QUALITY ATTRIBUTES (NFRs)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")

# Candidate rows
# Row 1: PE
ax.text(0.8, 3.25, r"$\mathbf{s}_{i,1}^{\mathrm{NFR}}$ : \"Response latency shall remain strictly below 200 ms\"", fontsize=9.5, color="#2c3e50")
ax.text(8.0, 3.25, r"$\rightarrow$ Class: $\mathbf{PE}$", fontsize=9.5, fontweight="bold", color="#7570b3",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#edeaf6", edgecolor="#7570b3", lw=0.8))

# Divider 1
ax.plot([0.8, 9.9], [3.05, 3.05], color="#ccd1d9", linestyle="--", lw=0.8)

# Row 2: SE
ax.text(0.8, 2.75, r"$\mathbf{s}_{i,2}^{\mathrm{NFR}}$ : \"Access tokens must be cryptographically hashed via SHA-256\"", fontsize=9.5, color="#2c3e50")
ax.text(8.0, 2.75, r"$\rightarrow$ Class: $\mathbf{SE}$", fontsize=9.5, fontweight="bold", color="#e7298a",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#faeaf3", edgecolor="#e7298a", lw=0.8))

# Divider 2
ax.plot([0.8, 9.9], [2.55, 2.55], color="#ccd1d9", linestyle="--", lw=0.8)

# Row 3: A
ax.text(0.8, 2.25, r"$\mathbf{s}_{i,3}^{\mathrm{NFR}}$ : \"Failover telemetry service shall guarantee 99.99% uptime\"", fontsize=9.5, color="#2c3e50")
ax.text(8.0, 2.25, r"$\rightarrow$ Class: $\mathbf{A}$", fontsize=9.5, fontweight="bold", color="#d95f02",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fbeee6", edgecolor="#d95f02", lw=0.8))

# Arrow Down 2
ax.annotate("", xy=(5.35, 1.45), xytext=(5.35, 1.9),
            arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#2c3e50", mutation_scale=15))
ax.text(5.5, 1.68, "Decoupled Cross-Stream Pairing", fontsize=8.5, color="#555555", fontstyle="italic")

# --- Step 3: Resulting Independent Training Instances ---
inst_box = FancyBboxPatch((0.5, 0.4), 9.7, 1.0, boxstyle="round,pad=0.1,rounding_size=0.15",
                          edgecolor="#8e44ad", facecolor="#f4ecf7", linewidth=1.2)
ax.add_patch(inst_box)

inst_tag = FancyBboxPatch((0.7, 1.15), 3.2, 0.32, boxstyle="round,pad=0.05,rounding_size=0.08",
                          edgecolor="#8e44ad", facecolor="#8e44ad")
ax.add_patch(inst_tag)
ax.text(2.3, 1.31, "DECOUPLED PAIR INSTANCES", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")

# 3 Distinct instance boxes
inst_colors = ["#7570b3", "#e7298a", "#d95f02"]
labels = [r"$(\mathbf{c}_i^{\mathrm{FR}}, \mathbf{s}_{i,1}^{\mathrm{NFR}}) \Rightarrow \mathbf{PE}$",
          r"$(\mathbf{c}_i^{\mathrm{FR}}, \mathbf{s}_{i,2}^{\mathrm{NFR}}) \Rightarrow \mathbf{SE}$",
          r"$(\mathbf{c}_i^{\mathrm{FR}}, \mathbf{s}_{i,3}^{\mathrm{NFR}}) \Rightarrow \mathbf{A}$"]

for idx, (lbl, col) in enumerate(zip(labels, inst_colors)):
    bx = FancyBboxPatch((0.8 + idx * 3.1, 0.52), 2.9, 0.58, boxstyle="round,pad=0.08,rounding_size=0.1",
                        edgecolor=col, facecolor="white", linewidth=1.1)
    ax.add_patch(bx)
    ax.text(0.8 + idx * 3.1 + 1.45, 0.81, lbl, ha="center", va="center", fontsize=10, fontweight="bold", color="#1a252f")

# ==============================================================================
# RIGHT PANEL: BENCHMARK IMPACT (BEFORE vs AFTER AT REPO SCALE)
# ==============================================================================
# Outer container Right
right_box = FancyBboxPatch((10.8, 0.2), 4.0, 5.8, boxstyle="round,pad=0.15,rounding_size=0.2",
                           edgecolor="#16a085", facecolor="#f8fbfb", linewidth=1.5, linestyle="--")
ax.add_patch(right_box)

ax.text(12.8, 5.75, "Benchmark Transformation", 
        ha="center", va="center", fontsize=12.5, fontweight="bold", color="#16a085")

# --- Top Box: Legacy Formulation ---
b_top = FancyBboxPatch((11.1, 3.4), 3.4, 2.0, boxstyle="round,pad=0.1,rounding_size=0.15",
                       edgecolor="#e67e22", facecolor="#fef5e7", linewidth=1.2)
ax.add_patch(b_top)

b_top_tag = FancyBboxPatch((11.3, 5.08), 3.0, 0.32, boxstyle="round,pad=0.05,rounding_size=0.08",
                           edgecolor="#e67e22", facecolor="#e67e22")
ax.add_patch(b_top_tag)
ax.text(12.8, 5.24, "LEGACY / ISOLATED SETUP", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")

ax.text(12.8, 4.65, "1 Functional Requirement", ha="center", fontsize=9.5, fontweight="bold", color="#2c3e50")
ax.text(12.8, 4.35, "Isolated Quality Constraints (No Link)", ha="center", fontsize=8.5, color="#7f8c8d")
ax.text(12.8, 3.85, "Zero Context Awareness\nContext Dilution in Single-Stream", ha="center", fontsize=8.5, color="#c0392b", fontweight="bold")

# Big Transition Arrow Down
ax.annotate("", xy=(12.8, 2.7), xytext=(12.8, 3.35),
            arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#16a085", mutation_scale=20))

# --- Bottom Box: Proposed Formulation ---
b_bot = FancyBboxPatch((11.1, 0.4), 3.4, 2.2, boxstyle="round,pad=0.1,rounding_size=0.15",
                       edgecolor="#27ae60", facecolor="#eafaf1", linewidth=1.2)
ax.add_patch(b_bot)

b_bot_tag = FancyBboxPatch((11.3, 2.28), 3.0, 0.32, boxstyle="round,pad=0.05,rounding_size=0.08",
                           edgecolor="#27ae60", facecolor="#27ae60")
ax.add_patch(b_bot_tag)
ax.text(12.8, 2.44, "PROPOSED RECOVERY SETUP", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white")

ax.text(12.8, 1.88, "208 Context-Paired Traces", ha="center", fontsize=10.5, fontweight="bold", color="#1e8449")
ax.text(12.8, 1.55, "P1-P15 Project Boundaries Preserved", ha="center", fontsize=8.5, color="#2c3e50")
ax.text(12.8, 1.25, "8,654 Multi-Corpus Generalization", ha="center", fontsize=8.5, color="#2c3e50")
ax.text(12.8, 0.75, "Zero-Leakage Project Splits\nGuarantees Unbiased Evaluation", ha="center", fontsize=8.5, color="#1e8449", fontweight="bold")

plt.tight_layout()

out_path = "C:/Users/umert/.gemini/antigravity/scratch/DualStream-NFR/figures/Figure_1_Dataset_Reformulation.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
artifact_path = "C:/Users/umert/.gemini/antigravity/brain/511ae778-3e5e-45a5-a8e3-e8f8b876420a/Figure_1_Dataset_Reformulation.png"
plt.savefig(artifact_path, dpi=300, bbox_inches="tight")
print("SUCCESS: Generalized Figure 1 rendered!")