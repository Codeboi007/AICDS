import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# --- Custom Dataset ---
class VideoDataset(Dataset):
    def __init__(self, root_dir, frames_per_video=16, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.samples = []

        for label, class_name in enumerate(sorted(os.listdir(root_dir))):  # 0: ai, 1: real
            class_dir = os.path.join(root_dir, class_name)
            for video_file in os.listdir(class_dir):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    path = os.path.join(class_dir, video_file)
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._extract_frames(video_path)
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames)  # [T, C, H, W]
        return frames, label

    def _extract_frames(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // self.frames_per_video)
        frames, count = [], 0

        while len(frames) < self.frames_per_video and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)
            count += 1

        cap.release()
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1])  # pad with last
        return frames

# --- CRNN Model with Pretrained ResNet ---
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

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Dataset + Loader ---
dataset = VideoDataset("training-data/video-data", transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

# --- Training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training Started...")
for epoch in range(1, 6):
    model.train()
    train_loss = 0.0
    for videos, labels in train_loader:
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={val_loss/len(val_loader):.4f}")

# --- Evaluation ---
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=["AI", "Real"]))
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

# --- Save Model ---
torch.save(model.state_dict(), "video_crnn_resnet_model.pth")
print("Model saved as video_crnn_resnet_model.pth")
