import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Model
# ----------------------------

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
model.to(device)
model.eval()

# ----------------------------
# Image Preprocessing
# ----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# IMPORTANT: Check your class order!
# It depends on folder names inside dataset/
# Example:
# dataset/
#    fake/
#    real/
# If fake comes first alphabetically, then:
# class_names = ['fake', 'real']

class_names = ['fake', 'real']   # ⚠️ Adjust if needed

def detect_face_and_draw_box(image_path):

    # Load image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # Draw rectangle on faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Save new image
    base, ext = os.path.splitext(image_path)
    boxed_path = base + "_boxed" + ext
    cv2.imwrite(boxed_path, image)

    return boxed_path


def predict_image(image_path):
    print("PREDICT FUNCTION STARTED")

    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None

            self.target_layer.register_forward_hook(self.forward_hook)
            self.target_layer.register_backward_hook(self.backward_hook)

        def forward_hook(self, module, input, output):
            self.activations = output

        def backward_hook(self, module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def generate(self):
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1)
            cam = torch.relu(cam)
            cam = cam.squeeze().detach().cpu().numpy()
            cam = cv2.normalize(cam, None, 0, 255, cv2.NORM_MINMAX)
            return cam.astype("uint8")

    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)

    confidence, predicted_class = torch.max(probs, 1)

    confidence = confidence.item() * 100
    predicted_class = predicted_class.item()

    label = class_names[predicted_class].capitalize()

    if confidence < 60:
        label = "Uncertain"
        explanation = "The model is not confident enough to make a reliable decision."
    elif label == "Real":
        explanation = "No strong manipulation artifacts detected in facial regions."
    else:
        explanation = "Potential synthetic textures or inconsistencies detected."

   
    # ----------------------------
# 🔥 Grad-CAM Heatmap (ResNet18)
# ----------------------------

    image_cv = cv2.imread(image_path)

# Create GradCAM object
    target_layer = model.layer4[1].conv2
    grad_cam = GradCAM(model, target_layer)

# Forward pass
    image_tensor.requires_grad = True
    outputs = model(image_tensor)
    score = outputs[0, predicted_class]

    model.zero_grad()
    score.backward()

# Generate CAM
    cam = grad_cam.generate()

# Resize to original image size
    cam = cv2.resize(cam, (image_cv.shape[1], image_cv.shape[0]))

# Apply TURBO colormap
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_TURBO)

# Overlay on original image
    overlay = cv2.addWeighted(image_cv, 0.6, heatmap, 0.4, 0)

    boxed_path = image_path.replace(".", "_heatmap.")
    cv2.imwrite(boxed_path, overlay)
    return label, round(confidence, 2), explanation, boxed_path
    print("RETURNING VALUES")