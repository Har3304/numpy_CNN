import os
import cv2
import numpy as np
import pickle
from glob import glob
from sklearn.model_selection import train_test_split

# -----------------------------
# Config
# -----------------------------
DATA_ROOT = r"F:/sm"        # expects subfolders: tri, circ, squ, rect
IMG_SIZE = 64
NUM_CLASSES = 4
LR = 0.01
EPOCHS = 10
BATCH_SIZE = 16
SEED = 42
np.random.seed(SEED)

LABEL_MAP = {"tri": 0, "circ": 1, "squ": 2, "rect": 3}

# -----------------------------
# Data loading
# -----------------------------
def load_dataset(data_root=DATA_ROOT, img_size=IMG_SIZE):
    X, y = [], []
    for name, label in LABEL_MAP.items():
        folder = os.path.join(data_root, name)
        files = sorted(glob(os.path.join(folder, "*.png")))
        for fp in files:
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            X.append(img[None, ...])  # (1, H, W) channel-first
            y.append(label)
    X = np.stack(X, axis=0)  # (N, 1, H, W)
    y = np.array(y, dtype=np.int64)
    return X, y

def one_hot(y, num_classes=NUM_CLASSES):
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out

# -----------------------------
# Layers (NumPy-only)
# -----------------------------
class Conv2D:
    """
    Naive convolution (NCHW) with stride=1, padding='valid'
    W: (F, C, KH, KW), b: (F,)
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        KH, KW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.W = 0.01 * np.random.randn(out_channels, in_channels, KH, KW).astype(np.float32)
        self.b = np.zeros((out_channels,), dtype=np.float32)

    def forward(self, x):
        self.x = x  # (N, C, H, W)
        N, C, H, W = x.shape
        F, _, KH, KW = self.W.shape
        out_H = H - KH + 1
        out_W = W - KW + 1
        out = np.zeros((N, F, out_H, out_W), dtype=np.float32)

        # naive loops
        for n in range(N):
            for f in range(F):
                for i in range(out_H):
                    for j in range(out_W):
                        region = x[n, :, i:i+KH, j:j+KW]  # (C, KH, KW)
                        out[n, f, i, j] = np.sum(region * self.W[f]) + self.b[f]
        self.cache_shape = out.shape
        return out

    def backward(self, dout, lr):
        x = self.x
        N, C, H, W = x.shape
        F, _, KH, KW = self.W.shape
        _, _, out_H, out_W = self.cache_shape

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dx = np.zeros_like(x)

        for n in range(N):
            for f in range(F):
                for i in range(out_H):
                    for j in range(out_W):
                        db[f] += dout[n, f, i, j]
                        region = x[n, :, i:i+KH, j:j+KW]
                        dW[f] += dout[n, f, i, j] * region
                        dx[n, :, i:i+KH, j:j+KW] += dout[n, f, i, j] * self.W[f]

        # SGD update
        self.W -= lr * dW
        self.b -= lr * db
        return dx

class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout, lr):
        return dout * self.mask

class MaxPool2x2:
    """
    2x2 max-pooling, stride=2, no overlap
    """
    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "H,W must be even for 2x2 pooling"
        out = x.reshape(N, C, H//2, 2, W//2, 2).max(axis=3).max(axis=4)

        # store argmax mask for backward
        self.mask = np.zeros_like(x, dtype=bool)
        for n in range(N):
            for c in range(C):
                for i in range(H//2):
                    for j in range(W//2):
                        block = x[n, c, 2*i:2*i+2, 2*j:2*j+2]
                        m = (block == np.max(block))
                        self.mask[n, c, 2*i:2*i+2, 2*j:2*j+2] = m
        return out

    def backward(self, dout, lr):
        N, C, H, W = self.x.shape
        dx = np.zeros_like(self.x)
        # distribute dout to max positions
        for n in range(N):
            for c in range(C):
                for i in range(H//2):
                    for j in range(W//2):
                        dx[n, c, 2*i:2*i+2, 2*j:2*j+2] += self.mask[n, c, 2*i:2*i+2, 2*j:2*j+2] * dout[n, c, i, j]
        return dx

class Flatten:
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout, lr):
        return dout.reshape(self.x_shape)

class Dense:
    def __init__(self, in_dim, out_dim):
        # He/Xavier-ish small init
        scale = np.sqrt(2.0 / in_dim)
        self.W = (scale * np.random.randn(in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros((out_dim,), dtype=np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout, lr):
        dW = self.x.T @ dout
        db = np.sum(dout, axis=0)
        dx = dout @ self.W.T
        # SGD
        self.W -= lr * dW
        self.b -= lr * db
        return dx

class SoftmaxCrossEntropy:
    def forward(self, logits, y_true_onehot):
        # logits: (N, K); y_true_onehot: (N, K)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        self.probs = probs
        self.y = y_true_onehot
        # cross-entropy
        N = logits.shape[0]
        loss = -np.sum(y_true_onehot * np.log(probs + 1e-12)) / N
        return loss, probs

    def backward(self):
        # dL/dlogits = (probs - y) / N
        N = self.y.shape[0]
        return (self.probs - self.y) / N

# -----------------------------
# Model definition (NumPy CNN)
# -----------------------------
class NumPyCNN:
    def __init__(self, img_size=IMG_SIZE, num_classes=NUM_CLASSES):
        # (N, 1, 64, 64)
        self.layers = [
            Conv2D(1, 8, 5),   # -> (N, 8, 60, 60)
            ReLU(),
            MaxPool2x2(),      # -> (N, 8, 30, 30)
            Conv2D(8, 16, 3),  # -> (N, 16, 28, 28)
            ReLU(),
            MaxPool2x2(),      # -> (N, 16, 14, 14)
            Flatten(),         # -> (N, 16*14*14)
            Dense(16*14*14, 64),
            ReLU(),
            Dense(64, num_classes)
        ]
        self.criterion = SoftmaxCrossEntropy()

    def forward(self, x, y_onehot):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        loss, probs = self.criterion.forward(out, y_onehot)
        return loss, probs, out

    def backward(self, lr):
        dout = self.criterion.backward()
        # backprop through layers in reverse
        for layer in reversed(self.layers):
            dout = layer.backward(dout, lr)

    def predict(self, x):
        out = x
        for layer in self.layers:
            # forward without storing caches—ok for eval here
            out = layer.forward(out)
        probs = np.exp(out - out.max(axis=1, keepdims=True))
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)
    def save(self, path="cnn_weights.pkl"):
        weights = []
        for layer in self.layers:
            state = {}
            if hasattr(layer, "W"):
                state["W"] = layer.W
            if hasattr(layer, "b"):
                state["b"] = layer.b
            weights.append(state)
        with open(path, "wb") as f:
            pickle.dump(weights, f)

    def load(self, path="cnn_weights.pkl"):
        with open(path, "rb") as f:
            weights = pickle.load(f)
        for layer, state in zip(self.layers, weights):
            if "W" in state:
                layer.W = state["W"]
            if "b" in state:
                layer.b = state["b"]
# -----------------------------
# Training utilities
# -----------------------------
def iterate_minibatches(X, Y, batch_size):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        sel = idx[i:i+batch_size]
        yield X[sel], Y[sel]

def accuracy(model, X, y):
    preds = model.predict(X)
    return (preds == y).mean()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    X, y = load_dataset()
    print("Loaded:", X.shape, y.shape)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    Y_train_oh = one_hot(y_train)
    Y_val_oh = one_hot(y_val)

    model = NumPyCNN()

    for ep in range(1, EPOCHS+1):
        # train
        losses = []
        for xb, yb in iterate_minibatches(X_train, Y_train_oh, BATCH_SIZE):
            loss, probs, logits = model.forward(xb, yb)
            model.backward(LR)
            losses.append(loss)

        # evaluate
        train_acc = accuracy(model, X_train, y_train)
        val_acc = accuracy(model, X_val, y_val)
        print(f"Epoch {ep:02d} | loss: {np.mean(losses):.4f} | train acc: {train_acc:.3f} | val acc: {val_acc:.3f}")
    model.save("cnn_weights.pkl")
    print("Weights saved.")
