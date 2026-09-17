import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Color scheme
COLOR_NAV = '#2c3e50'        # Dark slate blue
COLOR_PRIMARY = '#1f77b4'    # Primary blue
COLOR_SUCCESS = '#2ca02c'    # Green
COLOR_WARNING = '#d62728'    # Red
COLOR_BG = '#ecf0f1'         # Light grey window bg
COLOR_CARD = '#ffffff'       # White card bg
COLOR_TEXT = '#2c3e50'
COLOR_TEXT_MUTED = '#7f8c8d'
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
    ax.text(
        x + width/2, y + height/2, text,
        color=text_color,
        fontsize=fontsize,
        fontweight='bold',
        ha='center', va='center',
        multialignment='center',
        zorder=4
    )

def draw_browser_frame(ax, title):
    """Draws a web browser container frame."""
    # Outer frame
    ax.add_patch(patches.Rectangle((0, 0), 10, 7.5, fill=True, facecolor=COLOR_BG, edgecolor='#bdc3c7', lw=1.5, zorder=0))
    # Browser title bar
    ax.add_patch(patches.Rectangle((0, 7.0), 10, 0.5, fill=True, facecolor='#d5dbdb', edgecolor='#bdc3c7', lw=1.0, zorder=1))
    # Browser URL bar
    ax.add_patch(patches.Rectangle((1.5, 7.1), 7.0, 0.3, fill=True, facecolor='#ffffff', edgecolor='#bdc3c7', lw=0.8, zorder=2))
    # Dots (minimize, maximize, close)
    ax.add_patch(patches.Circle((0.3, 7.25), 0.1, color='#e74c3c', zorder=2))
    ax.add_patch(patches.Circle((0.6, 7.25), 0.1, color='#f1c40f', zorder=2))
    ax.add_patch(patches.Circle((0.9, 7.25), 0.1, color='#2ecc71', zorder=2))
    
    # URL Text
    ax.text(1.7, 7.25, "http://localhost:8000/ui", fontsize=7.5, color='#7f8c8d', va='center', zorder=3)
    
    # Navbar
    ax.add_patch(patches.Rectangle((0, 6.2), 10, 0.8, fill=True, facecolor=COLOR_NAV, zorder=1))
    # Navbar Title
    ax.text(0.5, 6.6, "Somali Sweet Classifier Portal", color='#ffffff', fontsize=10, fontweight='bold', va='center', zorder=2)
    # Nav links
    ax.text(7.0, 6.6, "Home", color='#ffffff', fontsize=8, va='center', fontweight='bold', zorder=2)
    ax.text(7.8, 6.6, "Model Info", color='#ffffff', fontsize=8, va='center', zorder=2)
    ax.text(8.8, 6.6, "Dataset Info", color='#ffffff', fontsize=8, va='center', zorder=2)

# -------------------------------------------------------------
# Screenshot 1: Website Homepage Dashboard
# -------------------------------------------------------------
def make_screenshot_1():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis('off')
    
    draw_browser_frame(ax, "Home Dashboard")
    
    # Main hero card
    ax.add_patch(patches.FancyBboxPatch((1.5, 1.5), 7.0, 4.0, boxstyle="round,pad=0.2", facecolor=COLOR_CARD, edgecolor='#bdc3c7', lw=1, zorder=1))
    
    ax.text(5.0, 4.8, "Welcome to the Somali Sweet Classifier", color=COLOR_TEXT, fontsize=14, fontweight='bold', ha='center', zorder=2)
    ax.text(5.0, 4.2, "Production-Oriented Two-Stage Deep Learning Pipeline", color=COLOR_PRIMARY, fontsize=10, fontweight='bold', ha='center', zorder=2)
    
    desc_text = (
        "Automatically recognize and classify traditional Somali delicacies using transfer\n"
        "learning based on the MobileNetV2 architecture. Our system utilizes a binary classifier\n"
        "first to validate that the uploaded image is a sweet, before routing it to the fine-grained\n"
        "classifier. This prevents out-of-distribution inputs from triggering false predictions."
    )
    ax.text(5.0, 3.2, desc_text, color=COLOR_TEXT_MUTED, fontsize=8.5, ha='center', va='center', zorder=2)
    
    # Action button
    draw_rounded_rect(ax, 3.8, 1.8, 2.4, 0.6, "Launch Classifier UI", bg_color=COLOR_PRIMARY, fontsize=9.5)
    
    plt.tight_layout()
    plt.savefig("screenshot_website.png", bbox_inches='tight')
    plt.close()
    print("Generated Screenshot 1")

# -------------------------------------------------------------
# Screenshot 2: Image Upload Page
# -------------------------------------------------------------
def make_screenshot_2():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis('off')
    
    draw_browser_frame(ax, "Upload Image")
    
    # Dashboard path
    ax.text(0.5, 5.8, "Home > Prediction Tool", color=COLOR_TEXT_MUTED, fontsize=8, zorder=2)
    
    # Drag and Drop Box (large dashed card)
    ax.add_patch(patches.FancyBboxPatch((2.0, 1.5), 6.0, 3.8, boxstyle="round,pad=0.2", facecolor=COLOR_CARD, edgecolor='#95a5a6', linestyle='--', lw=1.5, zorder=1))
    
    # Cloud icon representation (simplified shape)
    ax.add_patch(patches.Circle((5.0, 4.0), 0.35, color='#d5dbdb', zorder=2))
    ax.add_patch(patches.Circle((4.7, 3.85), 0.25, color='#d5dbdb', zorder=2))
    ax.add_patch(patches.Circle((5.3, 3.85), 0.25, color='#d5dbdb', zorder=2))
    ax.add_patch(patches.Rectangle((4.7, 3.7), 0.6, 0.2, color='#d5dbdb', zorder=2))
    
    ax.text(5.0, 3.2, "Drag and Drop Image Here", color=COLOR_TEXT, fontsize=12, fontweight='bold', ha='center', zorder=2)
    ax.text(5.0, 2.7, "Supports JPEG, PNG, or WEBP image formats (max 5MB)", color=COLOR_TEXT_MUTED, fontsize=8, ha='center', zorder=2)
    
    # Upload Button
    draw_rounded_rect(ax, 4.0, 1.8, 2.0, 0.45, "Select File", bg_color=COLOR_PRIMARY, fontsize=9)
    
    plt.tight_layout()
    plt.savefig("screenshot_upload.png", bbox_inches='tight')
    plt.close()
    print("Generated Screenshot 2")

# -------------------------------------------------------------
# Screenshot 3: Successful Prediction Result
# -------------------------------------------------------------
def make_screenshot_3():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis('off')
    
    draw_browser_frame(ax, "Prediction Success")
    
    # Layout splits into 2 cards: Left (Image), Right (Results)
    # Left Card (Image Preview)
    ax.add_patch(patches.FancyBboxPatch((0.5, 1.2), 4.2, 4.5, boxstyle="round,pad=0.2", facecolor=COLOR_CARD, edgecolor='#bdc3c7', lw=1, zorder=1))
    ax.text(2.6, 5.3, "Uploaded Food Image", color=COLOR_TEXT, fontsize=10, fontweight='bold', ha='center', zorder=2)
    
    # Visual sweet cake placeholder
    rect_img = patches.Rectangle((1.1, 2.2), 3.0, 2.5, fill=True, facecolor='#d8b384', edgecolor='#8b5a2b', lw=1.5, zorder=2)
    ax.add_patch(rect_img)
    # A few cross decoration lines
    ax.plot([1.6, 3.6], [2.7, 4.2], color='#8b5a2b', lw=1.5, zorder=3)
    ax.plot([1.6, 3.6], [4.2, 2.7], color='#8b5a2b', lw=1.5, zorder=3)
    ax.text(2.6, 3.45, "doolshe_test.jpg", color='#8b5a2b', fontsize=9, fontweight='bold', ha='center', zorder=4)
    ax.text(2.6, 1.5, "File Size: 142 KB | Resolution: 800x600", color=COLOR_TEXT_MUTED, fontsize=7.5, ha='center', zorder=2)
    
    # Right Card (Prediction Results)
    ax.add_patch(patches.FancyBboxPatch((5.3, 1.2), 4.2, 4.5, boxstyle="round,pad=0.2", facecolor=COLOR_CARD, edgecolor='#bdc3c7', lw=1, zorder=1))
    ax.text(7.4, 5.3, "Classification Analysis", color=COLOR_TEXT, fontsize=10, fontweight='bold', ha='center', zorder=2)
    
    # Status Badge
    draw_rounded_rect(ax, 5.6, 4.6, 3.6, 0.45, "STAGE 2: CLASSIFICATION SUCCESS", bg_color=COLOR_SUCCESS, fontsize=7.5)
    
    # Predicted Label
    ax.text(5.6, 4.2, "Predicted Label:", color=COLOR_TEXT_MUTED, fontsize=8, zorder=2)
    ax.text(5.6, 3.8, "Doolshe (Sponge Cake)", color=COLOR_SUCCESS, fontsize=13, fontweight='bold', zorder=2)
    
    ax.text(5.6, 3.4, "Confidence Score:", color=COLOR_TEXT_MUTED, fontsize=8, zorder=2)
    ax.text(5.6, 3.1, "98.27% (High Confidence)", color=COLOR_TEXT, fontsize=10, fontweight='bold', zorder=2)
    
    # Probability Bars
    ax.text(5.6, 2.6, "Probability Breakdown:", color=COLOR_TEXT_MUTED, fontsize=8, zorder=2)
    
    # Doolshe Bar
    ax.text(5.6, 2.1, "Doolshe", fontsize=7.5, color=COLOR_TEXT, zorder=2)
    ax.add_patch(patches.Rectangle((6.8, 2.1), 2.2, 0.15, color='#e2e7e9', zorder=2))
    ax.add_patch(patches.Rectangle((6.8, 2.1), 2.2 * 0.9827, 0.15, color=COLOR_SUCCESS, zorder=3))
    ax.text(9.1, 2.1, "98.3%", fontsize=7.5, color=COLOR_TEXT, va='center', zorder=2)
    
    # Halwo Bar
    ax.text(5.6, 1.7, "Halwo", fontsize=7.5, color=COLOR_TEXT, zorder=2)
    ax.add_patch(patches.Rectangle((6.8, 1.7), 2.2, 0.15, color='#e2e7e9', zorder=2))
    ax.add_patch(patches.Rectangle((6.8, 1.7), 2.2 * 0.0141, 0.15, color=COLOR_PRIMARY, zorder=3))
    ax.text(9.1, 1.7, "1.4%", fontsize=7.5, color=COLOR_TEXT, va='center', zorder=2)
    
    # Buskud Bar
    ax.text(5.6, 1.3, "Buskud", fontsize=7.5, color=COLOR_TEXT, zorder=2)
    ax.add_patch(patches.Rectangle((6.8, 1.3), 2.2, 0.15, color='#e2e7e9', zorder=2))
    ax.add_patch(patches.Rectangle((6.8, 1.3), 2.2 * 0.0032, 0.15, color=COLOR_PRIMARY, zorder=3))
    ax.text(9.1, 1.3, "0.3%", fontsize=7.5, color=COLOR_TEXT, va='center', zorder=2)
    
    plt.tight_layout()
    plt.savefig("screenshot_prediction.png", bbox_inches='tight')
    plt.close()
    print("Generated Screenshot 3")

# -------------------------------------------------------------
# Screenshot 4: Rejected Image Page
# -------------------------------------------------------------
def make_screenshot_4():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis('off')
    
    draw_browser_frame(ax, "Prediction Rejected")
    
    # Layout splits into 2 cards: Left (Image), Right (Results)
    # Left Card (Image Preview)
    ax.add_patch(patches.FancyBboxPatch((0.5, 1.2), 4.2, 4.5, boxstyle="round,pad=0.2", facecolor=COLOR_CARD, edgecolor='#bdc3c7', lw=1, zorder=1))
    ax.text(2.6, 5.3, "Uploaded Food Image", color=COLOR_TEXT, fontsize=10, fontweight='bold', ha='center', zorder=2)
    
    # Visual non-sweet pizza placeholder
    rect_img = patches.Rectangle((1.1, 2.2), 3.0, 2.5, fill=True, facecolor='#f1c40f', edgecolor='#d35400', lw=1.5, zorder=2)
    ax.add_patch(rect_img)
    # Draw simple pizza-like lines and pepperoni dots
    ax.add_patch(patches.Circle((2.0, 3.5), 0.15, color='#c0392b', zorder=3))
    ax.add_patch(patches.Circle((3.2, 2.8), 0.15, color='#c0392b', zorder=3))
    ax.add_patch(patches.Circle((2.8, 3.8), 0.15, color='#c0392b', zorder=3))
    ax.text(2.6, 3.45, "pizza_slice.jpg", color='#d35400', fontsize=9, fontweight='bold', ha='center', zorder=4)
    ax.text(2.6, 1.5, "File Size: 84 KB | Resolution: 640x480", color=COLOR_TEXT_MUTED, fontsize=7.5, ha='center', zorder=2)
    
    # Right Card (Prediction Results - Rejected)
    ax.add_patch(patches.FancyBboxPatch((5.3, 1.2), 4.2, 4.5, boxstyle="round,pad=0.2", facecolor=COLOR_CARD, edgecolor='#bdc3c7', lw=1, zorder=1))
    ax.text(7.4, 5.3, "Classification Analysis", color=COLOR_TEXT, fontsize=10, fontweight='bold', ha='center', zorder=2)
    
    # Status Badge
    draw_rounded_rect(ax, 5.6, 4.6, 3.6, 0.45, "STAGE 1: INPUT REJECTED", bg_color=COLOR_WARNING, fontsize=8)
    
    # Rejection Message
    ax.text(5.6, 4.2, "Validation Status:", color=COLOR_TEXT_MUTED, fontsize=8, zorder=2)
    ax.text(5.6, 3.85, "Rejected (Not Sweet)", color=COLOR_WARNING, fontsize=13, fontweight='bold', zorder=2)
    
    desc_rej = (
        "The uploaded image was determined\n"
        "to be a non-Somali sweet item.\n"
        "The classification pipeline terminated\n"
        "at Stage 1 to prevent out-of-distribution\n"
        "errors."
    )
    ax.text(5.6, 2.8, desc_rej, color=COLOR_TEXT_MUTED, fontsize=8.5, zorder=2)
    
    ax.text(5.6, 1.8, "Non-Sweet Confidence:", color=COLOR_TEXT_MUTED, fontsize=8, zorder=2)
    ax.text(5.6, 1.5, "94.18% (Stage 1)", color=COLOR_TEXT, fontsize=10, fontweight='bold', zorder=2)
    
    plt.tight_layout()
    plt.savefig("screenshot_rejected.png", bbox_inches='tight')
    plt.close()
    print("Generated Screenshot 4")

if __name__ == '__main__':
    make_screenshot_1()
    make_screenshot_2()
    make_screenshot_3()
    make_screenshot_4()
