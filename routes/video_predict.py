import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
from PIL import Image
from torchvision import transforms

# CRNN model with pretrained CNN (ResNet18)
class CRNN(nn.Module):
    def __init__(self, hidden_size=128):
        super(CRNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # remove last FC
        self.cnn = nn.Sequential(*modules)
        self.rnn = nn.GRU(input_size=512, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

        #Freeze CNN layers if desired
        for param in self.cnn.parameters():
           param.requires_grad = False  # comment out to fine-tune

    def forward(self, x):  #[B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        with torch.no_grad():
            features = self.cnn(x).view(B, T, -1)  #[B*T, 512, 1, 1] -> [B, T, 512]
        output, _ = self.rnn(features)
        return self.fc(output[:, -1, :])  # use last time step

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN().to(device)
model.load_state_dict(torch.load("models/video_crnn_resnet_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_frames(video_path, frames_per_video=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // frames_per_video)
    frames, count = [], 0

    while len(frames) < frames_per_video and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(transform(frame))
        count += 1
    cap.release()

    # pad if fewer frames
    while len(frames) < frames_per_video:
        frames.append(frames[-1])
    return torch.stack(frames)  # [T, C, H, W]

def predict_video(video_path):
    frames = extract_frames(video_path)
    frames = frames.unsqueeze(0).to(device)  # [1, T, C, H, W]
    with torch.no_grad():
        output = model(frames)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        label = "Real" if pred.item() == 1 else "AI"
        return label, round(conf.item() * 100, 2)
