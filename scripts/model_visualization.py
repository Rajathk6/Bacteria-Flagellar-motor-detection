import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch
from standalone_train import create_model_config

def add_box(ax, x, y, width, height, label, color='lightblue', alpha=0.3):
    rect = Rectangle((x, y), width, height, facecolor=color, alpha=alpha, edgecolor='black')
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', wrap=True)
    return x + width/2, y + height/2

def add_arrow(ax, start, end, color='black', alpha=0.5):
    arrow = FancyArrowPatch(start, end, arrowstyle='->', color=color, alpha=alpha,
                           connectionstyle='arc3,rad=0.2', mutation_scale=20)
    ax.add_patch(arrow)

def visualize_architecture():
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Encoder path (ResNet-200 backbone)
    encoder_boxes = [
        (0, 8, 2, 1, "Input\n(1 channel)"),
        (2, 8, 2, 1, "Conv1\n(64 channels)"),
        (4, 8, 2, 1, "Layer1\n(64 channels)"),
        (4, 6, 2, 1, "Layer2\n(128 channels)"),
        (4, 4, 2, 1, "Layer3\n(256 channels)"),
        (4, 2, 2, 1, "Layer4\n(512 channels)"),
    ]
    
    # Decoder path
    decoder_boxes = [
        (6, 2, 2, 1, "Decoder Block 1\n(256 channels)"),
        (6, 4, 2, 1, "Decoder Block 2\n(256 channels)"),
        (6, 6, 2, 1, "Decoder Block 3\n(256 channels)"),
        (6, 8, 2, 1, "Decoder Block 4\n(256 channels)"),
        (8, 8, 2, 1, "Segmentation Head\n(1 channel)")
    ]
    
    # Draw encoder boxes
    encoder_centers = []
    for x, y, w, h, label in encoder_boxes:
        center = add_box(ax, x, y, w, h, label, color='lightblue')
        encoder_centers.append(center)
    
    # Draw decoder boxes
    decoder_centers = []
    for x, y, w, h, label in decoder_boxes:
        center = add_box(ax, x, y, w, h, label, color='lightgreen')
        decoder_centers.append(center)
    
    # Draw skip connections
    for i in range(4):
        add_arrow(ax, encoder_centers[i+1], decoder_centers[3-i], color='red', alpha=0.3)
    
    # Draw main path arrows
    for i in range(len(encoder_centers)-1):
        add_arrow(ax, encoder_centers[i], encoder_centers[i+1])
    
    for i in range(len(decoder_centers)-1):
        add_arrow(ax, decoder_centers[i], decoder_centers[i+1])
    
    # Add title
    plt.title("3D UNet Architecture with ResNet-200 Backbone", pad=20)
    
    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.3, label='Encoder (ResNet-200)'),
        Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.3, label='Decoder'),
        FancyArrowPatch((0, 0), (1, 0), color='red', alpha=0.3, label='Skip Connections')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=3, fancybox=True, shadow=True)
    
    # Save the figure
    plt.savefig('model_architecture.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Model architecture visualization saved as 'model_architecture.png'")

if __name__ == "__main__":
    visualize_architecture()