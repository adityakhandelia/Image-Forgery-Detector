
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import io
def generate_ela_image(image: Image.Image, quality: int = 90) -> Image.Image:
    orig = image.convert("RGB")
    buffer = io.BytesIO()
    orig.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")
    ela_img = ImageChops.difference(orig, compressed)
    extrema = ela_img.getextrema()  # tuple per channel
    max_diff = max([ex[1] for ex in extrema]) if extrema else 0
    if max_diff == 0:
        # no difference -> return a near-black image (still RGB)
        return Image.new("RGB", orig.size)

    # Scale brightness so differences are visible
    scale = 255.0 / float(max_diff)
    ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)
    return ela_img.convert("RGB")
class VGG19_ELA(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super(VGG19_ELA, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ---------------------------
# Load Model
# ---------------------------
MODEL_PATH = "vgg_19.pth"

@st.cache_resource
def load_model(path: str = MODEL_PATH):
    model = VGG19_ELA(num_classes=2)
    try:
        state = torch.load(path, map_location="cpu")
    except Exception as e:
        st.error(f"Failed to load model file: {e}")
        return None
    if isinstance(state, dict):
        possible_keys = ["state_dict", "model_state_dict", "model", "net", "state"]
        found = False
        for k in possible_keys:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                found = True
                break

    # Remove common prefixes like "module." (from DataParallel) or "model."
    new_state = {}
    for k, v in state.items():
        new_key = k
        if k.startswith("module."):
            new_key = k[len("module."):]
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        new_state[new_key] = v

    # Try strict load first; if it fails, try with strict=False
    try:
        model.load_state_dict(new_state, strict=True)
    except Exception:
        try:
            model.load_state_dict(new_state, strict=False)
            st.warning("Loaded model with strict=False (some keys missing/mismatched).")
        except Exception as e:
            st.error(f"Failed to load weights into model: {e}")
            return None

    model.eval()
    return model

model = load_model()
if model is None:
    st.stop()  # stop the app if model couldn't be loaded

# ---------------------------
# Preprocess image for model
# ---------------------------
def preprocess_image(img: Image.Image, size: int = 128) -> torch.Tensor:
    """PIL Image -> torch tensor shape [1,3,H,W], float32 scaled to [0,1]."""
    img = img.convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0  # H,W,C
    arr = np.transpose(arr, (2, 0, 1))  # C,H,W
    tensor = torch.from_numpy(arr).unsqueeze(0)  # 1,C,H,W
    return tensor

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Image Forgery Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload a JPG/PNG image to analyze.")
else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Can't open uploaded image: {e}")
        st.stop()

    # Columns: original and ELA
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.write("Generating ELA image...")
        ela_image = generate_ela_image(image, quality=90)
        st.image(ela_image, caption="ELA Image", use_container_width=True)

    # Preprocess and predict
    try:
        img_tensor = preprocess_image(ela_image, size=128)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
    except Exception as e:
        st.error(f"Error during preprocessing/prediction: {e}")
        st.stop()

    # Interpret prediction
    # NOTE: Ensure your training label mapping matches this mapping:
    # e.g., 1 -> Authentic, 0 -> Forged (adjust if your training used different mapping)
    label_map = {1: "Authentic ✔", 0: "Forged ❌"}
    label = label_map.get(pred, f"Class {pred}")

    st.markdown("### Result")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {probs[pred]*100:.2f}%")
