"""
Phase 4 Part 3: Final Training
Executes the massive 3-day training run across all 474 stocks
with the winning hyperparameters from Part 2.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.config import Config
import importlib
train_model = importlib.import_module("train.04_train_card").train_model

def main():
    print("=======================================================")
    print("PHASE 4: FINAL MASSIVE TRADING MODEL TRAINING")
    print("=======================================================")
    print("WARNING: This will train on 474 stocks simultaneously.")
    print("It uses 0 MB RAM mmap SSD streaming and 0 num_workers.")
    print("Expected runtime: 2-3 Days")
    print("=======================================================\n")
    
    response = input("Are you sure you want to begin this 3-day run? (yes/no): ")
    if response.lower() not in ['y', 'yes']:
        print("Aborting.")
        return
        
    print("\nInitializing Weights & Biases (Offline Mode active if connection drops)...")
    
    try:
        # Run with the main Config class (Full 200 epochs)
        best_loss, best_acc = train_model(
            config_class=Config,
            run_name="card_phase4_final",
            use_wandb=True,
            wandb_project="card-stock-prediction" # Official project
        )
        
        print("\n" + "="*50)
        print("🎉 MASSIVE TRAINING RUN COMPLETE! 🎉")
        print("="*50)
        print(f"Final Best Loss: {best_loss:.6f}")
        print(f"Final Directional Accuracy: {best_acc:.2f}%")
        print("\nYour fully trained CARD weights are saved in checkpoints/phase4/")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR DURING TRAINING: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
