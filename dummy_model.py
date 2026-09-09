import os
import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 7
MODEL_SAVE_PATH = "model_weights/best_model.pth"

def create_dummy_model():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # Load EfficientNet-B0 with pretrained weights
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    
    # Modify the classifier head for 7 classes
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # Save the untrained model (just to test Streamlit)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Saved dummy model to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    create_dummy_model()
