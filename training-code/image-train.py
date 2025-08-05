from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from io import BytesIO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

# Load and split dataset
ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets")["train"]
df = ds.to_pandas()
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Custom dataset
class HuggingfaceImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_bytes = self.df.loc[idx, 'image']['bytes']
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        label = int(self.df.loc[idx, 'label'])

        if self.transform:
            image = self.transform(image)
        return image, label

# Data loaders
train_dataset = HuggingfaceImageDataset(train_df, transform=transform)
test_dataset = HuggingfaceImageDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)


# Basic CNN
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> [16, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))   # -> [32, 56, 56]
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BasicCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train and validate
num_epochs = 10
print("training start")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# Final evaluation
print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=["REAL", "FAKE"]))
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

# Save model
torch.save(model.state_dict(), "cnn_image_model.pth")
