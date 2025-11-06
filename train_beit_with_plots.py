import os
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import BeitForImageClassification, BeitImageProcessor
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc

# Collect logits and labels

# Device setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Dataset paths

train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"


# Processor and transforms

processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data   = datasets.ImageFolder(val_dir, transform=transform)
test_data  = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=4)
test_loader  = DataLoader(test_data, batch_size=4)

classes = train_data.classes
print("Classes:", classes)


# Load BEiT model

model = BeitForImageClassification.from_pretrained(
    "microsoft/beit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

# ============================================
# Training with history tracking
# ============================================
epochs = 5  # lightweight
train_losses = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images).logits
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Validation accuracy
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Accuracy={val_acc:.4f}")

# ============================================
# Plot Training Loss
# ============================================
plt.figure()
plt.plot(train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show(block=False)
plt.pause(2)


# ============================================
# Plot Validation Accuracy
# ============================================
plt.figure()
plt.plot(val_accuracies, marker='o', color='green')
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.show(block=False)
plt.pause(2)


# ============================================
# Final Test Evaluation
# ============================================
def collect_logits_and_labels(loader):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images).logits.cpu()
            all_logits.append(logits)
            all_labels.append(labels.cpu())
    return torch.cat(all_logits, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()

val_logits, y_val = collect_logits_and_labels(val_loader)
test_logits, y_test = collect_logits_and_labels(test_loader)

# Convert logits -> probabilities for pneumonia = index 1
val_probs = torch.softmax(torch.tensor(val_logits), dim=1)[:, 1].numpy()
test_probs = torch.softmax(torch.tensor(test_logits), dim=1)[:, 1].numpy()

# Hard predictions (argmax)
y_val_pred = np.argmax(val_logits, axis=1)
y_test_pred = np.argmax(test_logits, axis=1)

# =========================
# Classification Reports
# =========================
print("\nValidation Report")
print(classification_report(y_val, y_val_pred, target_names=classes))
print("Test Report")
print(classification_report(y_test, y_test_pred, target_names=classes))

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix (Test)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show(block=False); plt.pause(2)

# =========================
# ROC Curve (Val + Test)
# =========================
fpr_v, tpr_v, _ = roc_curve(y_val, val_probs)
fpr_t, tpr_t, _ = roc_curve(y_test, test_probs)
auc_v, auc_t = auc(fpr_v, tpr_v), auc(fpr_t, tpr_t)

plt.figure()
plt.plot(fpr_v, tpr_v, label=f"Val AUC={auc_v:.3f}")
plt.plot(fpr_t, tpr_t, label=f"Test AUC={auc_t:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show(block=False); plt.pause(2)

# =========================
# Save model
# =========================
torch.save(model.state_dict(), "beit_xray_model.pth")
print("Model saved as beit_xray_model.pth ✅")

model.save_pretrained("beit_finetuned_xray")
processor.save_pretrained("beit_finetuned_xray")
print("Full model and processor saved in 'beit_finetuned_xray/' ✅")