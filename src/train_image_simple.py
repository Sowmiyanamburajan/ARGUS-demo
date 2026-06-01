# src/train_image_simple.py  (very small example)
import torch, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm, torch.nn as nn, torch.optim as optim

data_dir = "datasets"
train_tf = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                               transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
val_tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

train_ds = datasets.ImageFolder(os.path.join(data_dir,"train"), transform=train_tf)
val_ds = datasets.ImageFolder(os.path.join(data_dir,"val"), transform=val_tf)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
opt = optim.AdamW(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(4):
    model.train()
    for imgs, labs in train_loader:
        imgs, labs = imgs.to(device), labs.to(device)
        opt.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labs)
        loss.backward(); opt.step()
    # save checkpoint
    state = {'model_state': model.state_dict(), 'epoch': epoch}
    torch.save(state, 'checkpoints/image_best.pth')
    print("Epoch",epoch,"saved.")
