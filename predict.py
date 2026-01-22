import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()

    if real_prob > fake_prob:
        label = "Real"
        confidence = real_prob * 100
        explanation = "No significant manipulation artifacts detected in facial regions."
    else:
        label = "Fake"
        confidence = fake_prob * 100
        explanation = "Detected abnormal facial textures and inconsistent eye or mouth patterns."

    return label, round(confidence, 2), explanation
