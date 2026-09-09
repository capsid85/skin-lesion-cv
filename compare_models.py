import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 7

def get_resnet50():
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model, "ResNet-50"

def get_mobilenet_v3():
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, NUM_CLASSES)
    return model, "MobileNetV3"

def get_efficientnet_b0():
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    return model, "EfficientNet-B0"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    print("Comparing Models (Architecture and Parameters)...")
    models_to_test = [get_resnet50(), get_mobilenet_v3(), get_efficientnet_b0()]
    
    for model, name in models_to_test:
        params = count_parameters(model)
        print(f"{name}: {params/1e6:.2f}M parameters")
    
    print("\nNote: Training logic to compare accuracy-vs-parameter trade-off is identical")
    print("to train.py. Swap the `get_model()` function in train.py to evaluate fully.")
