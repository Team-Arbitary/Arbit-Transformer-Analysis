"""
Quick test script to train and test the anomaly detection model
"""
import sys
sys.path.append('.')

from train import train_autoencoder
from test import test_model

if __name__ == '__main__':
    print("="*60)
    print("THERMAL TRANSFORMER ANOMALY DETECTION")
    print("="*60)
    
    # Train with fewer epochs for quick testing
    print("\n[1/2] Training AutoEncoder on normal images...")
    model, losses = train_autoencoder(
        data_root='Dataset',
        batch_size=4,
        epochs=20,  # Reduced for quick testing
        learning_rate=0.001,
        img_size=256
    )
    
    print("\n[2/2] Testing on faulty and normal images...")
    results = test_model(
        model_path='models/best_model.pth',
        data_root='Dataset',
        img_size=256,
        threshold=0.5,
        save_dir='results'
    )
    
    print("\n" + "="*60)
    print("COMPLETED!")
    print("="*60)
    print("\nCheck the 'results' folder for:")
    print("  - Visualization of anomaly detection")
    print("  - Segmentation masks")
    print("  - Error analysis plots")
