import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from pathlib import Path
import time

# =========================
# CONFIGURATION
# =========================
DATA_DIR = Path("/content")
OUTPUT_DIR = Path("/content/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model hyperparameters — from KA-Transformer paper (Zhu et al. 2025)
N_FEATURES   = 11    # your 11 retained features
N_TIMESTEPS  = 12    # 12 hours before onset
D_MODEL      = 64    # hidden dimension (paper uses 512, we use 64 for speed)
N_HEADS      = 8     # attention heads
N_LAYERS     = 4     # encoder layers
D_FF         = 256   # feed-forward dimension (paper uses 2048, we use 256)
KERNEL_SIZE  = 16    # kernel attention size (paper uses 64)
DROPOUT      = 0.2   # dropout rate
BATCH_SIZE   = 256   # batch size
EPOCHS       = 100   # training epochs
LR           = 1e-4  # learning rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# =========================
# PIECE 1 — INPUT PROJECTION
# =========================
class InputProjection(nn.Module):
    def __init__(self, n_features, d_model, dropout):
        super().__init__()
        # YOUR CODE HERE
        # You need:
        # 1. A linear layer that maps n_features → d_model
        self.linear = nn.Linear(n_features, d_model)
        # 2. A layer norm for stability 
        self.norm = nn.LayerNorm(d_model)
        # 3. A dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape coming in: [batch, 12, 11]
        # x shape going out: [batch, 12, 64]
        # YOUR CODE HERE
        # Apply linear → layer norm → dropout
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
        
        # Create position encoding matrix — shape [n_timesteps, d_model]
        pe = torch.zeros(n_timesteps, d_model)
        
        # Position indices [0, 1, 2, ..., 11] — shape [12, 1]
        position = torch.arange(0, n_timesteps).unsqueeze(1).float()
        
        # Scaling factors for sine/cosine waves
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-np.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: [1, n_timesteps, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a trainable parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: [batch, 12, 64]
        # Add position encoding to x
        # self.pe shape: [1, 12, 64] — broadcasts across batch
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
        self.d_head = d_model // n_heads  # 64 // 6 = 10 (per head dimension)
        
        # Learned projections for queries, keys, values
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Learned kernel parameter β — starts at 1.0
        self.beta = nn.Parameter(torch.ones(1))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x shape: [batch, 12, 64]
        batch, seq_len, d_model = x.shape
        
        # Step 1 — Project to Q, K, V
        Q = self.W_q(x)  # [batch, 12, 64]
        K = self.W_k(x)  # [batch, 12, 64]
        V = self.W_v(x)  # [batch, 12, 64]
        
        # Step 2 — Reshape for multi-head attention
        # Split d_model into n_heads × d_head
        Q = Q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # Shape now: [batch, n_heads, 12, d_head]
        
        # Step 3 — Compute RBF kernel similarity
        # ||xi - xj||² for all pairs of timesteps
        # Q shape: [batch, heads, seq, d_head]
        Q_expand = Q.unsqueeze(3)          # [batch, heads, seq, 1, d_head]
        K_expand = K.unsqueeze(2)          # [batch, heads, 1, seq, d_head]
        diff = Q_expand - K_expand         # [batch, heads, seq, seq, d_head]
        dist_sq = (diff ** 2).sum(-1)      # [batch, heads, seq, seq]
        
        # Apply RBF kernel: exp(-β × ||xi - xj||²)
        attn_weights = torch.exp(-self.beta * dist_sq)
        
        # Step 4 — Normalize attention weights
        attn_weights = attn_weights / (attn_weights.sum(-1, keepdim=True) + 1e-9)
        attn_weights = self.dropout(attn_weights)
        
        # Step 5 — Apply attention to values
        out = torch.matmul(attn_weights, V)  # [batch, heads, seq, d_head]
        
        # Step 6 — Reshape back and project
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch, seq_len, d_model)  # [batch, 12, 64]
        out = self.W_o(out)
        
        # Step 7 — Residual connection + layer norm
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
        # You need two things:
        # 1. A KernelAttention layer
        self.attn = KernelAttention(d_model, n_heads, dropout)
        # 2. A FeedForward layer
        self.ff = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x):
        # Step 1 — pass through kernel attention
        x = self.attn(x)
        # Step 2 — pass through feed forward
        x = self.ff(x)

        # Return result
        return x
    


# =========================
# PIECE 6 — FULL KA-TRANSFORMER
# =========================
class KATransformer(nn.Module):
    def __init__(self, n_features, n_timesteps, d_model, n_heads, 
                 n_layers, d_ff, dropout):
        super().__init__()
        
        # 1. Input projection
        self.projection = InputProjection(n_features, d_model, dropout)
        # 2. Position encoding
        self.encoding = PositionEncoding(d_model, n_timesteps, dropout)
        # 3. Stack of N_LAYERS encoder blocks — use nn.ModuleList
        self.encoders = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        # 4. Final linear classifier (d_model → 1)
        self.classifier = nn.Linear(d_model, 1)
        # 5. Sigmoid activation
        self.sigmoid = nn.Sigmoid()
   

    def forward(self, x):
        # 1. Input projection
        x = self.projection(x)
        # 2. Position encoding
        x = self.encoding(x)
        # 3. Pass through each encoder block
        for encoder in self.encoders:
             x = encoder(x)

        # 4. Global average pooling — x.mean(dim=1)
        x = x.mean(dim=1)
        # 5. Linear classifier
        x = self.classifier(x)
        # 6. Sigmoid
        x = self.sigmoid(x)

        # Return output 
        return x.squeeze(-1)  # shape [batch] with probabilities
    

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
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_test  = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test  = np.load(DATA_DIR / "y_test.npy")
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_test  = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test  = torch.FloatTensor(y_test)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test,  y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, y_test

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
# PIECE 10 — EVALUATION
# =========================
def evaluate(model, test_loader, y_test):
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            preds = model(X_batch)
            all_preds.extend(preds.cpu().numpy())
    
    all_preds = np.array(all_preds)
    pred_labels = (all_preds >= 0.5).astype(int)
    
    auroc = roc_auc_score(y_test, all_preds)
    f1    = f1_score(y_test, pred_labels, average='macro')
    
    print("\n--- KA-TRANSFORMER RESULTS ---")
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
    print("=" * 60)
    
    # Load data
    train_loader, test_loader, y_test = load_data()
    
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer with cosine annealing — from paper
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Training loop
    print("\nTraining...")
    best_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        loss = train(model, train_loader, optimizer, criterion)
        scheduler.step()
        
        if epoch % 10 == 0:
            elapsed = time.time() - start
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {loss:.4f} | Time: {elapsed:.1f}s")
        
        # Save best model
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
    
    print("\nTraining complete. Evaluating best model...")
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt"))
    auroc, f1, preds = evaluate(model, test_loader, y_test)
    
    # Save results
    np.save(OUTPUT_DIR / "test_predictions.npy", preds)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 60)