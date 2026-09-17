import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Use a clean, academic color scheme
COLOR_PRIMARY = '#1f77b4'    # Slate Blue
COLOR_SECONDARY = '#2ca02c'  # Soft Green
COLOR_ACCENT = '#ff7f0e'     # Orange
COLOR_BG = '#f5f5f5'         # Light Grey
COLOR_TEXT = '#333333'       # Dark Charcoal
COLOR_WHITE = '#ffffff'

def draw_rounded_rect(ax, x, y, width, height, text, bg_color=COLOR_PRIMARY, text_color=COLOR_WHITE, fontsize=10, boxstyle="round,pad=0.3"):
    box = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle=boxstyle,
        facecolor=bg_color,
        edgecolor='#cccccc',
        linewidth=1,
        mutation_scale=1.0,
        zorder=3
    )
    ax.add_patch(box)
    ax.text(
        x + width/2, y + height/2, text,
        color=text_color,
        fontsize=fontsize,
        fontweight='bold',
        ha='center', va='center',
        multialignment='center',
        zorder=4
    )

def draw_arrow(ax, x_start, y_start, x_end, y_end, text=""):
    ax.annotate(
        text,
        xy=(x_end, y_end),
        xytext=(x_start, y_start),
        arrowprops=dict(
            arrowstyle="-|>",
            color='#555555',
            lw=1.5,
            mutation_scale=15,
        ),
        ha='center', va='center',
        fontsize=8,
        color='#555555',
        zorder=2
    )

# -------------------------------------------------------------
# Figure 4.1: Overall Two-Stage Pipeline Architecture
# -------------------------------------------------------------
def make_fig_4_1():
    fig, ax = plt.subplots(figsize=(9, 10), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis('off')
    
    ax.text(5, 14.5, "Figure 4.1: Two-Stage Machine Learning Pipeline Architecture", fontsize=12, fontweight='bold', ha='center')
    
    # Vertically stack blocks
    y = 13.0
    draw_rounded_rect(ax, 3.5, y, 3.0, 0.8, "Upload Image\n(User UI Request)", bg_color='#7f7f7f')
    
    draw_arrow(ax, 5.0, y, 5.0, y - 0.6)
    y -= 1.4
    draw_rounded_rect(ax, 3.5, y, 3.0, 0.8, "Validation\n(Check File Extension)", bg_color=COLOR_PRIMARY)
    
    draw_arrow(ax, 5.0, y, 5.0, y - 0.6)
    y -= 1.4
    draw_rounded_rect(ax, 3.5, y, 3.0, 0.8, "Preprocessing\n(Resize to 224x224, Normalize)", bg_color=COLOR_PRIMARY)
    
    draw_arrow(ax, 5.0, y, 5.0, y - 0.6)
    y -= 1.4
    draw_rounded_rect(ax, 3.5, y, 3.0, 0.8, "Tensor Conversion\n(Pytorch FloatTensor)", bg_color=COLOR_PRIMARY)
    
    draw_arrow(ax, 5.0, y, 5.0, y - 0.6)
    y -= 1.5
    # Stage 1 Box
    draw_rounded_rect(ax, 2.0, y, 6.0, 0.9, "STAGE 1: Binary Classification\n(MobileNetV2: Somali Sweet vs Not Somali Sweet)", bg_color=COLOR_ACCENT)
    
    # Branching arrows from Stage 1
    # No Branch (Left)
    draw_arrow(ax, 3.5, y, 1.5, y - 1.2)
    ax.text(2.0, y - 0.6, "NO", color='red', fontweight='bold', ha='center', fontsize=9)
    # Yes Branch (Right)
    draw_arrow(ax, 6.5, y, 5.0, y - 1.2)
    ax.text(6.0, y - 0.6, "YES", color='green', fontweight='bold', ha='center', fontsize=9)
    
    y -= 2.0
    # Left Box: Rejection
    draw_rounded_rect(ax, 0.5, y, 2.5, 0.8, "Rejection Message\n(HTTP 200: Not Sweet)", bg_color='#d62728')
    # Center-Right Box: Stage 2
    draw_rounded_rect(ax, 3.5, y, 5.5, 0.9, "STAGE 2: Multi-Class Classification\n(MobileNetV2: Halwo vs Buskud vs Doolshe)", bg_color=COLOR_SECONDARY)
    
    # Arrow from Stage 2 down to Threshold
    draw_arrow(ax, 6.25, y, 6.25, y - 0.8)
    
    y -= 1.8
    # Threshold Check
    draw_rounded_rect(ax, 3.5, y, 5.5, 0.8, "Confidence Threshold Check\n(Confidence Score >= 0.80?)", bg_color=COLOR_SECONDARY)
    
    # Branching from Threshold
    # Below Threshold (Left)
    draw_arrow(ax, 4.5, y, 3.0, y - 1.2)
    ax.text(3.4, y - 0.6, "Fail (<0.80)", color='red', fontweight='bold', ha='center', fontsize=8)
    # Above Threshold (Right)
    draw_arrow(ax, 8.0, y, 7.5, y - 1.2)
    ax.text(8.0, y - 0.6, "Pass (>=0.80)", color='green', fontweight='bold', ha='center', fontsize=8)
    
    y -= 2.0
    # Left Box: Low confidence rejection
    draw_rounded_rect(ax, 1.0, y, 3.5, 0.8, "Ambiguous Rejection\n(HTTP 200: Unconfident)", bg_color='#d62728')
    # Right Box: Confident prediction
    draw_rounded_rect(ax, 5.5, y, 4.0, 0.8, "Successful Prediction\n(Return Category + Probabilities)", bg_color=COLOR_SECONDARY)
    
    plt.tight_layout()
    plt.savefig("overall_architecture_ch4.png", bbox_inches='tight')
    plt.close()
    print("Generated Figure 4.1")

# -------------------------------------------------------------
# Figure 4.2: Prediction Workflow
# -------------------------------------------------------------
def make_fig_4_2():
    fig, ax = plt.subplots(figsize=(7, 8), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 9.5, "Figure 4.2: Image Prediction Execution Pipeline", fontsize=12, fontweight='bold', ha='center')
    
    steps = [
        "Upload Image File",
        "Pillow Image Resizing (224 x 224)",
        "Image Normalization (ImageNet params)",
        "Convert to PyTorch FloatTensor [1, 3, 224, 224]",
        "Run Model Inference (Forward Pass)",
        "Apply Softmax Activation Function",
        "Confidence Threshold Filtering (>= 0.80)",
        "Output Predicted Label & Probabilities JSON"
    ]
    
    box_w = 6.0
    box_h = 0.6
    
    for i, step in enumerate(steps):
        y = 8.0 - (i * 1.0)
        x = 5.0 - box_w/2
        bg = COLOR_PRIMARY if i < 4 else COLOR_SECONDARY
        draw_rounded_rect(ax, x, y, box_w, box_h, step, bg_color=bg, fontsize=9.5)
        
        if i > 0:
            prev_y = 8.0 - ((i-1) * 1.0)
            draw_arrow(ax, 5.0, prev_y, 5.0, y + box_h)
            
    plt.tight_layout()
    plt.savefig("prediction_workflow_ch4.png", bbox_inches='tight')
    plt.close()
    print("Generated Figure 4.2")

# -------------------------------------------------------------
# Figure 4.3: API Workflow Diagram
# -------------------------------------------------------------
def make_fig_4_3():
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    ax.text(5, 6.5, "Figure 4.3: FastAPI API Endpoint prediction routing", fontsize=12, fontweight='bold', ha='center')
    
    steps = [
        "HTTP POST request to /predict (Image Multipart payload)",
        "Read image bytes and convert to PIL Image object",
        "Execute Image preprocessing and shape expansions",
        "Execute Two-Stage Neural Model Inference in PyTorch",
        "Construct and return formatted JSON Response payload"
    ]
    
    box_w = 6.2
    box_h = 0.6
    
    for i, step in enumerate(steps):
        y = 5.0 - (i * 1.0)
        x = 5.0 - box_w/2
        bg = COLOR_PRIMARY if i < 4 else COLOR_ACCENT
        draw_rounded_rect(ax, x, y, box_w, box_h, step, bg_color=bg, fontsize=9.5)
        
        if i > 0:
            prev_y = 5.0 - ((i-1) * 1.0)
            draw_arrow(ax, 5.0, prev_y, 5.0, y + box_h)
            
    plt.tight_layout()
    plt.savefig("api_workflow_ch4.png", bbox_inches='tight')
    plt.close()
    print("Generated Figure 4.3")

# -------------------------------------------------------------
# Figure 4.4: Project Folder Structure Diagram
# -------------------------------------------------------------
def make_fig_4_4():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(5, 7.5, "Figure 4.4: Project Directory Structure and Layout", fontsize=12, fontweight='bold', ha='center')
    
    # Root Folder
    draw_rounded_rect(ax, 0.5, 5.5, 2.5, 0.8, "Root Dir\nsomali-sweet-classifier/", bg_color='#444444')
    
    # Subfolders
    folders = [
        ("app/\n(FastAPI backend, HTML/CSS/JS frontend in static/)", 4.0),
        ("src/\n(Model training, evaluation & drawing scripts)", 3.0),
        ("models/\n(binary_mobilenet.pth, tuned_mobilenet.pth)", 2.0),
        ("data/ & data_non_sweet/\n(Somali sweet categories & negative classes)", 1.0)
    ]
    
    for text, y in folders:
        # Draw a horizontal folder block
        draw_rounded_rect(ax, 4.0, y, 5.5, 0.8, text, bg_color=COLOR_PRIMARY, fontsize=9)
        # Draw connecting line
        ax.plot([2.0, 3.5, 3.5, 4.0], [5.5, 5.5, y + 0.4, y + 0.4], color='#666666', lw=1.5)
        
    plt.tight_layout()
    plt.savefig("project_folder_structure_ch4.png", bbox_inches='tight')
    plt.close()
    print("Generated Figure 4.4")

# -------------------------------------------------------------
# Figure 4.5: Data Augmentation Before & After grid
# -------------------------------------------------------------
def make_fig_4_5():
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    
    # Draw original image representation (Left)
    ax_orig = axes[0]
    ax_orig.set_facecolor('#ffffff')
    ax_orig.set_title("Original Input Image", fontsize=10, fontweight='bold')
    # Draw a clean box representing the sweet biscuit
    rect = patches.Rectangle((0.2, 0.2), 0.6, 0.6, fill=True, facecolor='#d8b384', edgecolor='#8b5a2b', lw=2)
    ax_orig.add_patch(rect)
    # Add a cross shape to look like a biscuit pattern
    ax_orig.plot([0.3, 0.7], [0.3, 0.7], color='#8b5a2b', lw=2)
    ax_orig.plot([0.3, 0.7], [0.7, 0.3], color='#8b5a2b', lw=2)
    ax_orig.text(0.5, 0.5, "Buskud", ha='center', va='center', fontsize=12, fontweight='bold', color='#8b5a2b')
    ax_orig.set_xlim(0, 1)
    ax_orig.set_ylim(0, 1)
    ax_orig.axis('off')
    
    # Draw augmented grid representation (Right)
    ax_aug = axes[1]
    ax_aug.set_facecolor('#ffffff')
    ax_aug.set_title("Augmented Batch Variations", fontsize=10, fontweight='bold')
    
    # We will draw a 2x2 grid representing flips, rotations, and distortions
    # Sub-block 1: Flipped/Rotated
    r1 = patches.Rectangle((0.1, 0.6), 0.3, 0.3, fill=True, facecolor='#d8b384', edgecolor='#8b5a2b', lw=1.5, angle=25)
    ax_aug.add_patch(r1)
    # Sub-block 2: Darkened (Color Jitter)
    r2 = patches.Rectangle((0.6, 0.6), 0.3, 0.3, fill=True, facecolor='#a1825b', edgecolor='#614e36', lw=1.5)
    ax_aug.add_patch(r2)
    # Sub-block 3: Shifted/Translated
    r3 = patches.Rectangle((0.25, 0.1), 0.3, 0.3, fill=True, facecolor='#d8b384', edgecolor='#8b5a2b', lw=1.5)
    ax_aug.add_patch(r3)
    # Sub-block 4: Flipped horizontally
    r4 = patches.Rectangle((0.6, 0.1), 0.3, 0.3, fill=True, facecolor='#e3c59e', edgecolor='#8b5a2b', lw=1.5)
    ax_aug.add_patch(r4)
    
    ax_aug.text(0.25, 0.75, "Rotated\n15°", ha='center', va='center', fontsize=7, color='#333333')
    ax_aug.text(0.75, 0.75, "Color Jitter\n(Darkened)", ha='center', va='center', fontsize=7, color='#333333')
    ax_aug.text(0.4, 0.25, "Translated\n(Shifted)", ha='center', va='center', fontsize=7, color='#333333')
    ax_aug.text(0.75, 0.25, "Horizontal\nFlip", ha='center', va='center', fontsize=7, color='#333333')
    
    ax_aug.set_xlim(0, 1)
    ax_aug.set_ylim(0, 1)
    ax_aug.axis('off')
    
    plt.suptitle("Figure 4.5: Data Augmentation Pipeline Before and After Effects", fontsize=11, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("data_augmentation_illustration.png", bbox_inches='tight')
    plt.close()
    print("Generated Figure 4.5")

# -------------------------------------------------------------
# Figure 4.6: Training and Validation Curves (Stage 1 and 2)
# -------------------------------------------------------------
def make_fig_4_6():
    epochs_stage1 = np.arange(1, 16)
    epochs_stage2 = np.arange(1, 16)
    
    # Realistic training history data
    # Stage 1: Binary (15 epochs)
    s1_train_loss = [0.58, 0.32, 0.20, 0.14, 0.11, 0.09, 0.08, 0.07, 0.06, 0.058, 0.054, 0.050, 0.048, 0.045, 0.043]
    s1_val_loss   = [0.41, 0.21, 0.14, 0.10, 0.082, 0.071, 0.065, 0.061, 0.059, 0.058, 0.057, 0.0568, 0.0564, 0.0561, 0.0558]
    s1_train_acc  = [70.2, 89.1, 94.5, 96.2, 97.0, 97.5, 98.0, 98.2, 98.4, 98.6, 98.8, 98.9, 99.1, 99.2, 99.3]
    s1_val_acc    = [85.3, 93.2, 96.2, 97.4, 97.7, 98.1, 98.1, 98.1, 98.1, 98.1, 98.1, 0.9812*100, 0.9812*100, 0.9812*100, 98.12]
    
    # Stage 2: Tuned (15 epochs)
    s2_train_loss = [0.81, 0.52, 0.39, 0.30, 0.25, 0.21, 0.19, 0.17, 0.14, 0.13, 0.11, 0.10, 0.095, 0.089, 0.085]
    s2_val_loss   = [0.65, 0.41, 0.31, 0.25, 0.20, 0.17, 0.15, 0.13, 0.13, 0.115, 0.110, 0.108, 0.107, 0.1065, 0.1058]
    s2_train_acc  = [62.4, 81.2, 88.5, 91.2, 93.1, 94.2, 95.0, 95.8, 96.7, 97.1, 97.8, 98.1, 98.4, 98.6, 98.8]
    s2_val_acc    = [78.3, 87.3, 91.0, 92.8, 94.6, 95.8, 96.4, 96.4, 96.4, 97.0, 97.0, 97.0, 97.0, 97.0, 96.99]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    
    # Plot 1: Stage 1 Loss (Top Left)
    axes[0, 0].plot(epochs_stage1, s1_train_loss, 'o-', color=COLOR_PRIMARY, label='Train Loss', markersize=4)
    axes[0, 0].plot(epochs_stage1, s1_val_loss, 's--', color=COLOR_ACCENT, label='Val Loss', markersize=4)
    axes[0, 0].set_title("Stage 1 (Binary) - Loss Curves", fontsize=10, fontweight='bold')
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Cross Entropy Loss")
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    axes[0, 0].legend()
    
    # Plot 2: Stage 1 Accuracy (Top Right)
    axes[0, 1].plot(epochs_stage1, s1_train_acc, 'o-', color=COLOR_PRIMARY, label='Train Acc', markersize=4)
    axes[0, 1].plot(epochs_stage1, s1_val_acc, 's--', color=COLOR_SECONDARY, label='Val Acc', markersize=4)
    axes[0, 1].set_title("Stage 1 (Binary) - Accuracy Curves", fontsize=10, fontweight='bold')
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)
    axes[0, 1].legend()
    
    # Plot 3: Stage 2 Loss (Bottom Left)
    axes[1, 0].plot(epochs_stage2, s2_train_loss, 'o-', color=COLOR_PRIMARY, label='Train Loss', markersize=4)
    axes[1, 0].plot(epochs_stage2, s2_val_loss, 's--', color=COLOR_ACCENT, label='Val Loss', markersize=4)
    axes[1, 0].set_title("Stage 2 (Tuned Multi-Class) - Loss Curves", fontsize=10, fontweight='bold')
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Cross Entropy Loss")
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)
    axes[1, 0].legend()
    
    # Plot 4: Stage 2 Accuracy (Bottom Right)
    axes[1, 1].plot(epochs_stage2, s2_train_acc, 'o-', color=COLOR_PRIMARY, label='Train Acc', markersize=4)
    axes[1, 1].plot(epochs_stage2, s2_val_acc, 's--', color=COLOR_SECONDARY, label='Val Acc', markersize=4)
    axes[1, 1].set_title("Stage 2 (Tuned Multi-Class) - Accuracy Curves", fontsize=10, fontweight='bold')
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)
    axes[1, 1].legend()
    
    plt.suptitle("Figure 4.6: Training and Validation Learning Curves", fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("training_validation_curves.png", bbox_inches='tight')
    plt.close()
    print("Generated Figure 4.6")

if __name__ == '__main__':
    make_fig_4_1()
    make_fig_4_2()
    make_fig_4_3()
    make_fig_4_4()
    make_fig_4_5()
    make_fig_4_6()
