import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from pathlib import Path
import time

# =========================
# REPRODUCIBILITY
# =========================
# FIX: no seed was set anywhere before this — weight init and batch
# shuffling differed between every run, making before/after comparisons
# (e.g. alpha=0.25 vs alpha=0.6) partly confounded by run-to-run noise
# rather than purely the change being tested. Fixing the seed makes
# results reproducible and comparisons fair.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================
# CONFIGURATION
# =========================
DATA_DIR = Path("/content")
OUTPUT_DIR = Path("/content/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model hyperparameters — from KA-Transformer paper (Zhu et al. 2025)
N_FEATURES   = 21    # your 11 retained features
N_TIMESTEPS  = 12    # 12 hours before onset
D_MODEL      = 64    # hidden dimension (paper uses 512, we use 64 for speed)
N_HEADS      = 8     # attention heads
N_LAYERS     = 4     # encoder layers
D_FF         = 256   # feed-forward dimension (paper uses 2048, we use 256)
KERNEL_SIZE  = 16    # kernel attention size (paper uses 64) — NOTE: currently unused, see open issues
DROPOUT      = 0.2   # dropout rate
BATCH_SIZE   = 256   # batch size
EPOCHS       = 100   # training epochs (upper bound — early stopping may end training sooner)
LR           = 1e-4  # learning rate
VAL_SIZE     = 0.2   # fraction of TRAIN set carved out for validation

# EARLY STOPPING: stop training if validation loss hasn't improved for this
# many consecutive epochs. Training logs showed train loss still dropping
# at epoch 100 while val loss plateaus around epoch 50-60 — a classic
# overfitting signature. Checkpoint selection was already by best val loss
# (see FIX below), so this doesn't change which weights get used; it just
# stops wasting compute on epochs that can't produce a better checkpoint,
# and tells us how long training actually needs.
PATIENCE     = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# =========================
# PIECE 1 — INPUT PROJECTION
# =========================
class InputProjection(nn.Module):
    def __init__(self, n_features, d_model, dropout):
        super().__init__()
        self.linear = nn.Linear(n_features, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


# =========================
# PIECE 2 — POSITION ENCODING
# =========================
class PositionEncoding(nn.Module):
    def __init__(self, d_model, n_timesteps, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(n_timesteps, d_model)
        position = torch.arange(0, n_timesteps).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        x = self.dropout(x)
        return x


# =========================
# PIECE 3 — KERNEL ATTENTION
# =========================
class KernelAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.beta = nn.Parameter(torch.ones(1))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        Q_expand = Q.unsqueeze(3)
        K_expand = K.unsqueeze(2)
        diff = Q_expand - K_expand
        dist_sq = (diff ** 2).sum(-1)

        attn_weights = torch.exp(-self.beta * dist_sq)
        attn_weights = attn_weights / (attn_weights.sum(-1, keepdim=True) + 1e-9)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch, seq_len, d_model)
        out = self.W_o(out)

        out = self.norm(x + out)
        return out


# =========================
# PIECE 4 — FEED FORWARD NETWORK
# =========================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm(residual + x)
        x = self.dropout(x)
        return x


# =========================
# PIECE 5 — ENCODER BLOCK
# =========================
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = KernelAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = self.attn(x)
        x = self.ff(x)
        return x


# =========================
# PIECE 6 — FULL KA-TRANSFORMER
# =========================
class KATransformer(nn.Module):
    def __init__(self, n_features, n_timesteps, d_model, n_heads,
                 n_layers, d_ff, dropout):
        super().__init__()

        self.projection = InputProjection(n_features, d_model, dropout)
        self.encoding = PositionEncoding(d_model, n_timesteps, dropout)
        self.encoders = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.projection(x)
        x = self.encoding(x)
        for encoder in self.encoders:
            x = encoder(x)

        x = x.mean(dim=1)
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x.squeeze(-1)


# =========================
# PIECE 7 — FOCAL LOSS
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        bce = F.binary_cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-bce)

        # alpha for positive class (sepsis), 1-alpha for negative class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_weight = alpha_t * (1 - pt) ** self.gamma
        loss = focal_weight * bce
        return loss.mean()


# =========================
# PIECE 8 — DATA LOADING
# =========================
def load_data():
    X_train_full = np.load(DATA_DIR / "X_train.npy")
    X_test       = np.load(DATA_DIR / "X_test.npy")
    y_train_full = np.load(DATA_DIR / "y_train.npy")
    y_test       = np.load(DATA_DIR / "y_test.npy")

    # FIX: carve a validation split out of TRAIN (test set stays fully
    # held out, touched only once at the very end). Validation is used
    # for (a) model selection — saving the checkpoint with the best
    # validation loss, not the best training loss — and (b) picking the
    # decision threshold, instead of sweeping the threshold on the test
    # set itself, which was a soft form of test-set leakage.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=y_train_full
    )

    X_train = torch.FloatTensor(X_train)
    X_val   = torch.FloatTensor(X_val)
    X_test  = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_val   = torch.FloatTensor(y_val)
    y_test  = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset   = TensorDataset(X_val, y_val)
    test_dataset  = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {X_train.shape} | Sepsis: {y_train.sum():.0f} ({y_train.mean()*100:.1f}%)")
    print(f"Val:   {X_val.shape} | Sepsis: {y_val.sum():.0f} ({y_val.mean()*100:.1f}%)")
    print(f"Test:  {X_test.shape} | Sepsis: {y_test.sum():.0f} ({y_test.mean()*100:.1f}%)")

    return train_loader, val_loader, test_loader, y_val, y_test


# =========================
# PIECE 9 — TRAINING LOOP
# =========================
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


# =========================
# PIECE 9b — VALIDATION LOSS (for model selection)
# =========================
def compute_val_loss(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item()
    return total_loss / len(val_loader)


# =========================
# PIECE 9c — GET RAW PREDICTIONS (used for both val threshold search and test eval)
# =========================
def get_predictions(model, loader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch)
            all_preds.extend(preds.cpu().numpy())
    return np.array(all_preds)


# =========================
# PIECE 9d — FIND BEST THRESHOLD (on VALIDATION set only)
# =========================
def find_best_threshold(val_preds, y_val):
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        pred_labels = (val_preds >= thresh).astype(int)
        f1 = f1_score(y_val, pred_labels, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


# =========================
# PIECE 10 — EVALUATION (on TEST set, using a threshold chosen from VAL)
# =========================
def evaluate(model, test_loader, y_test, threshold):
    all_preds = get_predictions(model, test_loader)
    pred_labels = (all_preds >= threshold).astype(int)

    auroc = roc_auc_score(y_test, all_preds)
    f1    = f1_score(y_test, pred_labels, average='macro')

    print(f"\n--- KA-TRANSFORMER RESULTS (threshold={threshold:.2f}, chosen on validation set) ---")
    print(f"AUROC: {auroc:.4f}")
    print(f"F1 (macro): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, pred_labels,
          target_names=["No Sepsis", "Sepsis"]))

    return auroc, f1, all_preds


# =========================
# PIECE 11 — MAIN
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("KA-TRANSFORMER TRAINING")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print("=" * 60)

    # Load data (train/val/test — val carved out of train)
    train_loader, val_loader, test_loader, y_val, y_test = load_data()

    # Initialize model
    model = KATransformer(
        n_features  = N_FEATURES,
        n_timesteps = N_TIMESTEPS,
        d_model     = D_MODEL,
        n_heads     = N_HEADS,
        n_layers    = N_LAYERS,
        d_ff        = D_FF,
        dropout     = DROPOUT
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # NOTE: T_max stays at EPOCHS (the upper bound), not tied to early
    # stopping. CosineAnnealingLR computes the LR for a given epoch from
    # that epoch's index and T_max alone — it doesn't depend on whether
    # training later stops early. So the LR trajectory up to whatever
    # epoch ends up being "best" is unaffected by early stopping; we're
    # just skipping epochs afterward that couldn't produce a better
    # checkpoint anyway.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    criterion = FocalLoss(alpha=0.6, gamma=2.0)

    print("\nTraining...")
    best_val_loss = float('inf')

    # EARLY STOPPING: counts consecutive epochs with no val loss improvement.
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = compute_val_loss(model, val_loader, criterion)
        scheduler.step()

        if epoch % 10 == 0:
            elapsed = time.time() - start
            print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

        # FIX: model selection now uses VALIDATION loss, not training loss.
        # Previously the checkpoint with the lowest training loss was kept,
        # which has no signal about overfitting — on ~3,861 training
        # patients this risked saving a checkpoint that looked good on
        # train but had already started overfitting.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no val loss improvement for {PATIENCE} consecutive epochs).")
            break

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    print("Loading best model (by validation loss)...")

    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt"))

    # Pick decision threshold using VALIDATION predictions only —
    # the test set is not touched until the final evaluate() call below.
    val_preds = get_predictions(model, val_loader)
    best_thresh, val_f1_at_thresh = find_best_threshold(val_preds, y_val)
    print(f"\nBest threshold (selected on validation set): {best_thresh:.2f} "
          f"(validation F1 macro at this threshold: {val_f1_at_thresh:.4f})")

    # Final, single evaluation on the held-out test set
    auroc, f1, preds = evaluate(model, test_loader, y_test, threshold=best_thresh)

    np.save(OUTPUT_DIR / "test_predictions.npy", preds)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 60)