# train_mlp.py
import os, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

DATA_CSV = "data/gestures.csv"
OUT_DIR  = Path("models")

class GestureDataset(Dataset):
    def __init__(self, df):
        self.X = df[[c for c in df.columns if c.startswith("f")]].values.astype(np.float32)
        labels = df["label"].values.tolist()
        self.classes = sorted(list(set(labels)))
        self.cls_to_idx = {c:i for i,c in enumerate(self.classes)}
        self.y = np.array([self.cls_to_idx[c] for c in labels], dtype=np.int64)

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

class MLP(nn.Module):
    def __init__(self, in_dim=42, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(32, n_classes),
        )
    def forward(self, x): return self.net(x)

def main():
    assert os.path.exists(DATA_CSV), "Run data collection first."
    df = pd.read_csv(DATA_CSV)
    # shuffle
    df_train, df_val = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
        shuffle=True,   # train_test_split shuffles by default; keep for clarity
    )

    # (optional) tidy indices
    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)

    train_ds = GestureDataset(df_train)
    val_ds   = GestureDataset(df_val)

    # Ensure same class order across train/val
    classes = sorted(list(set(train_ds.classes) | set(val_ds.classes)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR/"classes.txt", "w") as f:
        for c in classes: f.write(c+"\n")

    # Remap val dataset to the same index space
    val_ds.cls_to_idx = {c:i for i,c in enumerate(classes)}
    val_ds.y = np.array([val_ds.cls_to_idx[c] for c in df_val["label"]], dtype=np.int64)
    train_ds.cls_to_idx = val_ds.cls_to_idx
    train_ds.y = np.array([train_ds.cls_to_idx[c] for c in df_train["label"]], dtype=np.int64)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=train_ds.X.shape[1], n_classes=len(classes)).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = 10; bad = 0

    for epoch in range(100):
        # train
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total += xb.size(0)

        # val
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)
                preds.extend(logits.argmax(1).cpu().numpy().tolist())
                gts.extend(yb.cpu().numpy().tolist())

        tr_acc  = tr_correct / max(1,tr_total)
        val_acc = val_correct / max(1,val_total)

        print(f"Epoch {epoch+1:03d} | train loss {tr_loss/tr_total:.4f} acc {tr_acc:.3f} | "
              f"val loss {val_loss/val_total:.4f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUT_DIR/"gesture_mlp.pt")
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    # Final report on val
    with torch.no_grad():
        model.load_state_dict(torch.load(OUT_DIR/"gesture_mlp.pt", map_location=device))
        model.eval()
        preds, gts = [], []
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            gts.extend(yb.numpy().tolist())
    inv = {i:c for i,c in enumerate(classes)}
    preds_lbl = [inv[i] for i in preds]
    gts_lbl   = [inv[i] for i in gts]
    try:
        print(classification_report(gts_lbl, preds_lbl, digits=3))
    except Exception:
        pass
    print(f"Classes: {classes}")
    print(f"Saved: {OUT_DIR/'gesture_mlp.pt'} and {OUT_DIR/'classes.txt'}")

if __name__ == "__main__":
    main()
