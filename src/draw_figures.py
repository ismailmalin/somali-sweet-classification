import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Use a clean, modern color scheme (Slate Blue, Teal, Light Grey)
COLOR_PRIMARY = '#1f77b4'    # Slate Blue
COLOR_SECONDARY = '#2ca02c'  # Soft Green
COLOR_ACCENT = '#ff7f0e'     # Orange
COLOR_BG = '#f5f5f5'         # Light Grey
COLOR_TEXT = '#333333'       # Dark Charcoal
COLOR_WHITE = '#ffffff'

def draw_rounded_rect(ax, x, y, width, height, text, bg_color=COLOR_PRIMARY, text_color=COLOR_WHITE, fontsize=10, boxstyle="round,pad=0.3"):
    """Helper to draw a rounded box with text."""
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
    
    # Center text in box
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
    """Helper to draw a clean arrow between points."""
    ax.annotate(
        text,
        xy=(x_end, y_end),
        xytext=(x_start, y_start),
        arrowprops=dict(
            arrowstyle="-|>",
            color='#555555',
            lw=1.5,
            mutation_scale=15,
            patchA=None, patchB=None,
            shrinkA=0, shrinkB=0
        ),
        ha='center', va='center',
        fontsize=8,
        color='#555555',
        zorder=2
    )

# -------------------------------------------------------------
# Figure 3.1: Data Collection & Cleaning Pipeline
# -------------------------------------------------------------
def make_fig_3_1():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, "Figure 3.1: Data Collection & Cleaning Pipeline", fontsize=12, fontweight='bold', ha='center')
    
    # Define boxes vertically
    boxes = [
        ("Data Collection\n(Search Engines, Social Media, Blogs)\nRaw count: 839 images", 9.5),
        ("Format Validation & File Inspection\n(Remove corrupted/unreadable files)", 7.5),
        ("Duplicate Detection & Removal\n(MD5 hashing: 12 duplicates removed)", 5.5),
        ("Data Preprocessing & Normalization\n(Resize to 224x224, ImageNet normalization)", 3.5),
        ("Dataset Splitting\n(80% Train: 661 images | 20% Val: 166 images)", 1.5)
    ]
    
    box_w = 6.0
    box_h = 1.0
    
    for i, (text, y) in enumerate(boxes):
        x = 5.0 - box_w/2
        # Use different colors for input and output stages
        bg = COLOR_PRIMARY if i < 4 else COLOR_SECONDARY
        draw_rounded_rect(ax, x, y, box_w, box_h, text, bg_color=bg)
        
        # Connect arrows
        if i > 0:
            prev_y = boxes[i-1][1]
            draw_arrow(ax, 5.0, prev_y, 5.0, y + box_h)
            
    plt.tight_layout()
    plt.savefig("data_collection_pipeline.png", bbox_inches='tight')
    plt.close()
    print("Generated Figure 3.1")

# -------------------------------------------------------------
# Figure 3.2: MobileNetV2 Architecture Diagram
# -------------------------------------------------------------
def make_fig_3_2():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, "Figure 3.2: MobileNetV2 Transfer Learning Architecture", fontsize=12, fontweight='bold', ha='center')
    
    # Draw boxes block-wise horizontally
    # 1. Input Image
    draw_rounded_rect(ax, 0.5, 3.0, 1.8, 1.5, "Input Image\n(224 x 224 x 3)", bg_color='#7f7f7f')
    draw_arrow(ax, 2.3, 3.75, 2.8, 3.75)
    
    # 2. Frozen Backbone
    draw_rounded_rect(ax, 2.8, 2.0, 4.0, 3.5, "MobileNetV2 Backbone\n(Pre-trained on ImageNet)\n\nBlocks 0 - 14: Frozen\nBlocks 15 - 17: Fine-Tuned", bg_color=COLOR_PRIMARY)
    draw_arrow(ax, 6.8, 3.75, 7.3, 3.75)
    
    # 3. Classifier Head (Trained)
    draw_rounded_rect(ax, 7.3, 2.2, 2.2, 3.1, "Classifier Head\n(Newly Added)\n\n- Dropout (0.2)\n- Linear (1280 to 3)\n- Softmax", bg_color=COLOR_SECONDARY)
    draw_arrow(ax, 9.5, 3.75, 10.0, 3.75)
    
    # 4. Predictions
    draw_rounded_rect(ax, 10.0, 2.8, 1.5, 1.9, "Prediction\n- Xalwo\n- Buskud\n- Doolshe", bg_color=COLOR_ACCENT)
    
    plt.tight_layout()
    plt.savefig("mobilenetv2_architecture.png", bbox_inches='tight')
    plt.close()
    print("Generated Figure 3.2")

# -------------------------------------------------------------
# Figure 3.3: Training & Fine-Tuning Workflow
# -------------------------------------------------------------
def make_fig_3_3():
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, "Figure 3.3: Model Training and Fine-Tuning Workflow", fontsize=12, fontweight='bold', ha='center')
    
    # Flowchart boxes
    y = 10.0
    draw_rounded_rect(ax, 1.5, y, 7.0, 0.8, "Step 1: Compute Class Weights (Inversely proportional to frequency)", bg_color=COLOR_PRIMARY)
    
    draw_arrow(ax, 5.0, y, 5.0, y - 0.7)
    y -= 1.5
    draw_rounded_rect(ax, 1.5, y, 7.0, 0.8, "Step 2: Initialize MobileNetV2 with ImageNet Weights & Freeze Feature Layers", bg_color=COLOR_PRIMARY)
    
    draw_arrow(ax, 5.0, y, 5.0, y - 0.7)
    y -= 1.5
    draw_rounded_rect(ax, 1.5, y, 7.0, 0.8, "Step 3: Train Classifier Head (10 Epochs, Adam LR = 0.001)", bg_color=COLOR_PRIMARY)
    
    draw_arrow(ax, 5.0, y, 5.0, y - 0.7)
    y -= 1.5
    draw_rounded_rect(ax, 1.5, y, 7.0, 0.8, "Step 4: Unfreeze Last 3 blocks of Backbone Features (Blocks 15-17)", bg_color=COLOR_SECONDARY)
    
    draw_arrow(ax, 5.0, y, 5.0, y - 0.7)
    y -= 1.5
    draw_rounded_rect(ax, 1.5, y, 7.0, 0.8, "Step 5: Perform Fine-Tuning (15 Epochs, Adam LR = 0.0001 + Plateau Scheduler)", bg_color=COLOR_SECONDARY)
    
    draw_arrow(ax, 5.0, y, 5.0, y - 0.7)
    y -= 1.5
    draw_rounded_rect(ax, 1.5, y, 7.0, 0.8, "Step 6: Evaluate Best Model Checkpoint & Save State Dictionary", bg_color=COLOR_ACCENT)
    
    plt.tight_layout()
    plt.savefig("training_workflow.png", bbox_inches='tight')
    plt.close()
    print("Generated Figure 3.3")

# -------------------------------------------------------------
# Figure 3.4: FastAPI System Architecture
# -------------------------------------------------------------
def make_fig_3_4():
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 9.5, "Figure 3.4: FastAPI System Architecture", fontsize=12, fontweight='bold', ha='center')
    
    # Client Side (Frontend)
    rect_client = patches.Rectangle((0.5, 1.0), 3.5, 7.0, fill=True, facecolor=COLOR_BG, edgecolor='#aaaaaa', zorder=1)
    ax.add_patch(rect_client)
    ax.text(2.25, 7.6, "CLIENT (Web Browser UI)\nHTML5 + CSS3 + JS", fontsize=10, fontweight='bold', ha='center', color=COLOR_TEXT)
    
    draw_rounded_rect(ax, 0.8, 5.0, 2.9, 1.0, "Interactive UI Form\n(Drag & drop image upload)", bg_color=COLOR_PRIMARY)
    draw_rounded_rect(ax, 0.8, 2.0, 2.9, 1.0, "Result Rendering Engine\n(Display labels & confidence)", bg_color=COLOR_PRIMARY)
    
    # Server Side (Backend)
    rect_server = patches.Rectangle((6.0, 1.0), 3.5, 7.0, fill=True, facecolor=COLOR_BG, edgecolor='#aaaaaa', zorder=1)
    ax.add_patch(rect_server)
    ax.text(7.75, 7.6, "SERVER (FastAPI Backend)\nPython + PyTorch", fontsize=10, fontweight='bold', ha='center', color=COLOR_TEXT)
    
    draw_rounded_rect(ax, 6.3, 5.5, 2.9, 0.8, "API Router (/predict)", bg_color=COLOR_SECONDARY)
    draw_rounded_rect(ax, 6.3, 4.0, 2.9, 0.8, "Image Preprocessing\n(Resize to 224, Normalize)", bg_color=COLOR_SECONDARY)
    draw_rounded_rect(ax, 6.3, 2.5, 2.9, 0.8, "MobileNetV2 Inference\n(Tuned PyTorch Model)", bg_color=COLOR_SECONDARY)
    draw_rounded_rect(ax, 6.3, 1.2, 2.9, 0.8, "JSON Response Generator\n(Predict probabilities)", bg_color=COLOR_SECONDARY)
    
    # Communication arrows
    # Upload Arrow
    ax.annotate(
        "HTTP POST\n/predict (Image File)",
        xy=(6.0, 5.9),
        xytext=(4.0, 5.9),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_ACCENT, lw=2, mutation_scale=15),
        ha='center', va='center', fontsize=8, color=COLOR_TEXT, fontweight='bold'
    )
    
    # Response Arrow
    ax.annotate(
        "HTTP 200 OK\n(JSON Results)",
        xy=(4.0, 2.5),
        xytext=(6.0, 2.5),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_ACCENT, lw=2, mutation_scale=15),
        ha='center', va='center', fontsize=8, color=COLOR_TEXT, fontweight='bold'
    )
    
    # Internal Flow Server
    draw_arrow(ax, 7.75, 5.5, 7.75, 4.8)
    draw_arrow(ax, 7.75, 4.0, 7.75, 3.3)
    draw_arrow(ax, 7.75, 2.5, 7.75, 2.0)
    
    plt.tight_layout()
    plt.savefig("fastapi_architecture.png", bbox_inches='tight')
    plt.close()
    print("Generated Figure 3.4")

# -------------------------------------------------------------
# Figure 3.5: Sequence Diagram
# -------------------------------------------------------------
def make_fig_3_5():
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(5, 9.5, "Figure 3.5: Client-Server Image Prediction Sequence Flow", fontsize=12, fontweight='bold', ha='center')
    
    # Define vertical lifelines
    lifelines = [
        ("User", 1.5),
        ("Web Browser UI", 4.0),
        ("FastAPI Server", 6.5),
        ("PyTorch Model", 8.5)
    ]
    
    # Draw vertical lines
    for name, x in lifelines:
        ax.plot([x, x], [1.0, 8.5], linestyle='--', color='#999999', lw=1.5, zorder=1)
        # Header box
        draw_rounded_rect(ax, x - 0.8, 8.5, 1.6, 0.5, name, bg_color='#444444', fontsize=9)
        # Footer box
        draw_rounded_rect(ax, x - 0.8, 0.5, 1.6, 0.5, name, bg_color='#444444', fontsize=9)
        
    # Sequence interactions
    y = 7.7
    # 1. User uploads image and clicks predict
    ax.annotate("", xy=(4.0, y), xytext=(1.5, y), arrowprops=dict(arrowstyle="->", color=COLOR_TEXT, lw=1.2))
    ax.text(2.75, y + 0.1, "1. Selects image & submits form", fontsize=8, ha='center')
    
    # 2. Browser sends HTTP POST to FastAPI
    y -= 1.0
    ax.annotate("", xy=(6.5, y), xytext=(4.0, y), arrowprops=dict(arrowstyle="->", color=COLOR_PRIMARY, lw=1.5))
    ax.text(5.25, y + 0.1, "2. POST /predict (Image multipart)", fontsize=8, ha='center', color=COLOR_PRIMARY, fontweight='bold')
    
    # 3. FastAPI preprocesses image
    y -= 1.0
    ax.annotate("", xy=(6.8, y - 0.4), xytext=(6.5, y), arrowprops=dict(arrowstyle="->", connectionstyle="bar,fraction=0.3", color=COLOR_SECONDARY, lw=1.2))
    ax.text(7.6, y - 0.2, "3. Resizes (224x224)\n    & normalizes tensor", fontsize=7, ha='left', va='center')
    
    # 4. FastAPI calls PyTorch Model
    y -= 1.2
    ax.annotate("", xy=(8.5, y), xytext=(6.5, y), arrowprops=dict(arrowstyle="->", color=COLOR_PRIMARY, lw=1.5))
    ax.text(7.5, y + 0.1, "4. model(tensor)", fontsize=8, ha='center')
    
    # 5. Model returns predictions
    y -= 1.0
    ax.annotate("", xy=(6.5, y), xytext=(8.5, y), arrowprops=dict(arrowstyle="->", color='#999999', linestyle='--', lw=1.2))
    ax.text(7.5, y + 0.1, "5. Returns raw logits", fontsize=8, ha='center', color='#666666')
    
    # 6. FastAPI computes Softmax and packages JSON
    y -= 1.0
    ax.annotate("", xy=(6.8, y - 0.4), xytext=(6.5, y), arrowprops=dict(arrowstyle="->", connectionstyle="bar,fraction=0.3", color=COLOR_SECONDARY, lw=1.2))
    ax.text(7.6, y - 0.2, "6. Computes probabilities\n    & builds JSON response", fontsize=7, ha='left', va='center')
    
    # 7. FastAPI returns JSON to Browser
    y -= 1.2
    ax.annotate("", xy=(4.0, y), xytext=(6.5, y), arrowprops=dict(arrowstyle="->", color=COLOR_PRIMARY, linestyle='--', lw=1.5))
    ax.text(5.25, y + 0.1, "7. JSON response (probabilities)", fontsize=8, ha='center', color=COLOR_PRIMARY, fontweight='bold')
    
    # 8. Browser displays result to User
    y -= 0.8
    ax.annotate("", xy=(1.5, y), xytext=(4.0, y), arrowprops=dict(arrowstyle="->", color=COLOR_TEXT, lw=1.2))
    ax.text(2.75, y + 0.1, "8. Renders classification result", fontsize=8, ha='center')
    
    plt.tight_layout()
    plt.savefig("prediction_sequence.png", bbox_inches='tight')
    plt.close()
    print("Generated Figure 3.5")

if __name__ == '__main__':
    make_fig_3_1()
    make_fig_3_2()
    make_fig_3_3()
    make_fig_3_4()
    make_fig_3_5()
