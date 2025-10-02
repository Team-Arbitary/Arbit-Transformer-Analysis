import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from dataset import ThermalDataset
from model import AnomalyAutoEncoder


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize tensor for visualization"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def generate_anomaly_mask(anomaly_map, threshold=0.5):
    """
    Convert anomaly heatmap to binary segmentation mask
    
    Args:
        anomaly_map: numpy array of shape (H, W) with values in [0, 1]
        threshold: threshold value for binarization
    
    Returns:
        Binary mask (H, W)
    """
    # Normalize to [0, 1]
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    
    # Apply threshold
    mask = (anomaly_map > threshold).astype(np.uint8) * 255
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def visualize_results(original, reconstructed, anomaly_map, mask, save_path=None):
    """
    Visualize original image, reconstruction, anomaly map, and segmentation mask
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Reconstructed image
    axes[1].imshow(reconstructed)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    
    # Anomaly heatmap
    im = axes[2].imshow(anomaly_map, cmap='jet')
    axes[2].set_title('Anomaly Heatmap')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    # Segmentation mask overlay
    overlay = original.copy()
    if len(mask.shape) == 2:
        mask_colored = np.zeros_like(original)
        mask_colored[:, :, 0] = mask  # Red channel for anomalies
        overlay = cv2.addWeighted(original, 0.7, mask_colored, 0.3, 0)
    
    axes[3].imshow(overlay)
    axes[3].set_title('Segmentation Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def test_model(
    model_path='models/best_model.pth',
    data_root='Dataset',
    img_size=256,
    threshold=0.5,
    save_dir='results',
    device=None
):
    """
    Test the trained model and generate anomaly segmentations
    """
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = AnomalyAutoEncoder(in_channels=3, latent_dim=128)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test dataset
    print("Loading test data...")
    test_dataset = ThermalDataset(
        root_dir=data_root,
        mode='test',
        img_size=img_size
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    print(f"\nTesting on {len(test_dataset)} images...")
    
    # Store results
    results = {
        'normal': {'errors': [], 'paths': []},
        'faulty': {'errors': [], 'paths': []}
    }
    
    # Process each image
    with torch.no_grad():
        for idx, (image, label, img_path) in enumerate(tqdm(test_loader, desc="Processing images")):
            image = image.to(device)
            
            # Get anomaly map and reconstruction
            anomaly_map, reconstructed = model.get_anomaly_map(image)
            
            # Calculate reconstruction error (average over all pixels)
            recon_error = F.mse_loss(reconstructed, image).item()
            
            # Convert to numpy for visualization
            image_np = image.cpu().squeeze(0).permute(1, 2, 0).numpy()
            image_np = denormalize(torch.tensor(image_np).permute(2, 0, 1)).permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
            
            reconstructed_np = reconstructed.cpu().squeeze(0).permute(1, 2, 0).numpy()
            reconstructed_np = denormalize(torch.tensor(reconstructed_np).permute(2, 0, 1)).permute(1, 2, 0).numpy()
            reconstructed_np = (reconstructed_np * 255).astype(np.uint8)
            
            anomaly_map_np = anomaly_map.cpu().squeeze().numpy()
            
            # Generate segmentation mask
            mask = generate_anomaly_mask(anomaly_map_np, threshold=threshold)
            
            # Store results
            label_str = 'faulty' if label.item() == 1 else 'normal'
            results[label_str]['errors'].append(recon_error)
            results[label_str]['paths'].append(img_path[0])
            
            # Save visualization for sample images (first 5 faulty and first 5 normal)
            if (label.item() == 1 and len([p for p in results['faulty']['paths']]) <= 5) or \
               (label.item() == 0 and len([p for p in results['normal']['paths']]) <= 5):
                
                filename = Path(img_path[0]).stem
                save_path = save_dir / f'{filename}_result.png'
                
                visualize_results(
                    image_np,
                    reconstructed_np,
                    anomaly_map_np,
                    mask,
                    save_path=save_path
                )
                
                # Also save the mask separately
                cv2.imwrite(str(save_dir / f'{filename}_mask.png'), mask)
    
    # Calculate statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    normal_errors = results['normal']['errors']
    faulty_errors = results['faulty']['errors']
    
    print(f"\nNormal Images (n={len(normal_errors)}):")
    print(f"  Mean reconstruction error: {np.mean(normal_errors):.6f}")
    print(f"  Std reconstruction error:  {np.std(normal_errors):.6f}")
    print(f"  Min error: {np.min(normal_errors):.6f}")
    print(f"  Max error: {np.max(normal_errors):.6f}")
    
    print(f"\nFaulty Images (n={len(faulty_errors)}):")
    print(f"  Mean reconstruction error: {np.mean(faulty_errors):.6f}")
    print(f"  Std reconstruction error:  {np.std(faulty_errors):.6f}")
    print(f"  Min error: {np.min(faulty_errors):.6f}")
    print(f"  Max error: {np.max(faulty_errors):.6f}")
    
    # Plot error distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(normal_errors, bins=20, alpha=0.7, label='Normal', color='blue')
    plt.hist(faulty_errors, bins=20, alpha=0.7, label='Faulty', color='red')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([normal_errors, faulty_errors], labels=['Normal', 'Faulty'])
    plt.ylabel('Reconstruction Error')
    plt.title('Error Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'error_analysis.png', dpi=150)
    plt.show()
    
    print(f"\nResults saved in: {save_dir}")
    
    return results


if __name__ == '__main__':
    # Test the model
    results = test_model(
        model_path='models/best_model.pth',
        data_root='Dataset',
        img_size=256,
        threshold=0.5,
        save_dir='results'
    )
