"""
Utility script to test anomaly detection on a single image
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import AnomalyAutoEncoder


def load_model(model_path, device='cpu'):
    """Load trained model"""
    model = AnomalyAutoEncoder(in_channels=3, latent_dim=128)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, img_size=256):
    """Load and preprocess image"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image.copy()
    
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0)
    
    return image_tensor, original


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize tensor for visualization"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def generate_anomaly_mask(anomaly_map, threshold=0.5):
    """Generate binary segmentation mask"""
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    mask = (anomaly_map > threshold).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def visualize_single_image(original, reconstructed, anomaly_map, mask, recon_error, save_path=None):
    """Visualize results for a single image"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Reconstructed image
    axes[0, 1].imshow(reconstructed)
    axes[0, 1].set_title('Reconstructed', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Difference
    diff = np.abs(original.astype(float) - reconstructed.astype(float))
    axes[0, 2].imshow(diff.astype(np.uint8))
    axes[0, 2].set_title('Difference', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Anomaly heatmap
    im = axes[1, 0].imshow(anomaly_map, cmap='jet')
    axes[1, 0].set_title(f'Anomaly Heatmap\nError: {recon_error:.4f}', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    
    # Binary mask
    axes[1, 1].imshow(mask, cmap='gray')
    axes[1, 1].set_title('Segmentation Mask', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Overlay
    overlay = original.copy()
    if len(mask.shape) == 2:
        mask_colored = np.zeros_like(original)
        mask_colored[:, :, 0] = mask
        overlay = cv2.addWeighted(original, 0.7, mask_colored, 0.3, 0)
    
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Anomaly Overlay', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def process_single_image(image_path, model_path='models/best_model.pth', threshold=0.5, save_dir=None):
    """Process a single image and visualize results"""
    
    print(f"Processing: {image_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    
    # Load and preprocess image
    print("Loading image...")
    image_tensor, original = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Get anomaly map
    print("Generating anomaly map...")
    with torch.no_grad():
        anomaly_map, reconstructed = model.get_anomaly_map(image_tensor)
        recon_error = torch.nn.functional.mse_loss(reconstructed, image_tensor).item()
    
    # Convert to numpy
    image_np = image_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    image_np = denormalize(torch.tensor(image_np).permute(2, 0, 1)).permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    
    reconstructed_np = reconstructed.cpu().squeeze(0).permute(1, 2, 0).numpy()
    reconstructed_np = denormalize(torch.tensor(reconstructed_np).permute(2, 0, 1)).permute(1, 2, 0).numpy()
    reconstructed_np = (reconstructed_np * 255).astype(np.uint8)
    
    anomaly_map_np = anomaly_map.cpu().squeeze().numpy()
    
    # Generate mask
    print("Generating segmentation mask...")
    mask = generate_anomaly_mask(anomaly_map_np, threshold=threshold)
    
    # Prepare save path
    save_path = None
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(image_path).stem
        save_path = save_dir / f'{filename}_analysis.png'
        
        # Also save mask
        cv2.imwrite(str(save_dir / f'{filename}_mask.png'), mask)
        print(f"Saved mask to: {save_dir / f'{filename}_mask.png'}")
    
    # Visualize
    print("\n" + "="*60)
    print(f"Reconstruction Error: {recon_error:.6f}")
    print(f"Anomaly Threshold: {threshold}")
    print("="*60)
    
    visualize_single_image(image_np, reconstructed_np, anomaly_map_np, mask, recon_error, save_path)
    
    return recon_error, mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test anomaly detection on a single image')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Path to trained model')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary mask (0-1)')
    parser.add_argument('--save-dir', type=str, default='single_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    process_single_image(
        image_path=args.image_path,
        model_path=args.model,
        threshold=args.threshold,
        save_dir=args.save_dir
    )
