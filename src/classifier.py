import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

CLASS_NAMES = [
    "Disturbed Galaxy",
    "Merging Galaxy",
    "Round Smooth Galaxy",
    "In-between Round Smooth Galaxy",
    "Cigar Shaped Smooth Galaxy",
    "Barred Spiral Galaxy",
    "Unbarred Tight Spiral Galaxy",
    "Unbarred Loose Spiral Galaxy",
    "Edge-on Galaxy without Bulge",
    "Edge-on Galaxy with Bulge"
]

def load_model(model_path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(image: Image.Image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
    top3 = [
        (CLASS_NAMES[top3_indices[0][i].item()], round(top3_probs[0][i].item() * 100, 2))
        for i in range(3)
    ]

    return CLASS_NAMES[predicted.item()], round(confidence.item() * 100, 2), top3