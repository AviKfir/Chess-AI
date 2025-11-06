"""
cnn_model.py

Convolutional Neural Network (CNN) for chess move policy prediction.
Runs the policy head, masks out illegal moves using move_to_index(move) encoding,
re-normalizes probabilities over the legal set, and then either selects the argmax legal move.

Architecture:
-------------
Input:
    - A batch of chess board encodings of shape (B, 13, 8, 8)
      where:
        • B = batch size (number of samples processed in parallel).
        • 13 = number of input channels (6 piece types × 2 colors + side-to-move plane).
        • 8×8 = chess board grid.

Output:
    - A batch of logits of shape (B, OUTPUT_DIM)
      where OUTPUT_DIM = 4672, corresponding to the fixed move-encoding scheme:
        • 3584 queen-like moves   (64 squares × 8 directions × 7 steps)
        • 512 knight-like moves   (64 squares × 8 offsets)
        • 576 under-promotions     (64 squares × 3 directions × 3 piece types: N, B, R)
      Total = 4672 possible move slots.

These logits are not probabilities yet; they can be passed through softmax
(while masking illegal moves) to obtain a probability distribution over legal moves.

"""

from features import OUTPUT_DIM  # Number of outputs in the fixed move encoding space (4672)
from typing import Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
import features
import chess


class ChessModel(nn.Module):
    """
    CNN policy head for chess move prediction.

    Forward pass:
        x -> Conv2d(13→64) -> ReLU
          -> Conv2d(64→128) -> ReLU
          -> Flatten
          -> Linear(8192→256) -> ReLU
          -> Linear(256→OUTPUT_DIM)

    Input:
        x: Tensor of shape (B, 13, 8, 8)
           where B = batch size (number of samples in the mini-batch).

    Output:
        Tensor of shape (B, OUTPUT_DIM)
        Each row is a vector of raw logits over all 4672 possible move slots.
    """

    def __init__(self, num_classes: int = OUTPUT_DIM):
        super().__init__()

        # First convolutional layer: 13 input planes -> 64 feature maps
        # Use bias=False when followed by BatchNorm
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Second convolutional layer: 64 -> 128 feature maps
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        # Flatten (B, 128, 8, 8) -> (B, 8192)
        self.flatten = nn.Flatten()

        # Fully connected hidden layer: 8192 -> 256
        self.fc1 = nn.Linear(8 * 8 * 128, 256, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(256)

        # Output layer: 256 -> OUTPUT_DIM (4672)
        self.fc2 = nn.Linear(256, num_classes)

        # Non-linear activation
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): input tensor of shape (B, 13, 8, 8),
                              where B = batch size.

        Returns:
            torch.Tensor: logits of shape (B, OUTPUT_DIM).
                          Each row corresponds to one position’s
                          un-normalized scores for all possible moves.
        """
        x = self.relu(self.bn1(self.conv1(x)))  # (B, 64, 8, 8)
        x = self.relu(self.bn2(self.conv2(x)))  # (B, 128, 8, 8)
        x = self.flatten(x)  # (B, 8192)
        x = self.relu(self.bn_fc1(self.fc1(x)))  # (B, 256)
        x = self.fc2(x)  # (B, 4672)
        return x


# ----------


def _softmax_np(x: np.ndarray) -> np.ndarray:
    """
    Numerically-stable softmax over a 1D vector (logits).
    Subtracts max(x) to avoid overflow in exp.

    Args:
        x: (N,) array of logits (may include very large negative numbers
           for masked-out entries).

    Returns:
        probs: (N,) float32 array with sum ≈ 1.0 (zeros on masked entries).
    """
    # Shift by max for numerical stability
    x = x - np.max(x)
    ex = np.exp(x, dtype=np.float64)
    s = ex.sum()
    if s <= 0.0 or not np.isfinite(s):
        # Fallback: all mass to zero-vector if something degenerate happens
        return np.zeros_like(x, dtype=np.float32)
    return (ex / s).astype(np.float32)


@torch.no_grad()
def predict_masked(
        model: torch.nn.Module,
        board: chess.Board,
        device: torch.device = torch.device("cpu"),
        temperature: Optional[float] = None,
        sample: bool = False,
        return_probs: bool = False,
) -> Tuple[Optional[chess.Move], Optional[np.ndarray]]:
    """
    Run the CNN policy, mask out illegal moves, renormalize, and pick a move.

    Pipeline:
      1) Encode board -> (1,13,8,8) tensor on device
      2) Forward pass -> logits (1, OUTPUT_DIM)
      3) Build mask for legal moves -> (OUTPUT_DIM,) bool
      4) Set illegal logits to a large negative (approx. -inf)
      5) Optional temperature scaling (divide logits by temperature)
      6) Softmax -> probabilities over legal moves only
      7) Select: argmax (sample=False) or sample by probs (sample=True)

    Args:
        model:       Your ChessModel producing (B, OUTPUT_DIM) logits.
        board:       python-chess Board for the current position.
        device:      "cpu" or "cuda" device for the forward pass.
        temperature: If None or 1.0, plain softmax. If >1 → flatter; <1 → sharper.
        sample:      If False, choose highest-prob legal move. If True, sample.
        return_probs:If True, also return the (OUTPUT_DIM,) numpy probs vector.

    Returns:
        (move, probs):
            move:  chess.Move (or None if no legal moves).
            probs: (OUTPUT_DIM,) float32 numpy array if return_probs=True, else None.
    """
    # 1) Encode board to planes (13,8,8) -> (1,13,8,8)
    planes = features.board_to_13_planes(board)
    x = torch.from_numpy(planes).unsqueeze(0).to(device)

    # 2) Forward pass to get logits over the fixed move vocabulary
    model.eval()  # ensures eval behavior for Dropout/BatchNorm etc.
    logits_t = model(x)  # shape: (1, OUTPUT_DIM)
    logits = logits_t.squeeze(0).cpu().numpy()  # -> (OUTPUT_DIM,)

    # 3) Legal mask from your encoding: True at indices that correspond to legal moves
    mask = features.legal_policy_mask(board)  # shape: (OUTPUT_DIM,)

    # 4) Apply mask by assigning ~-inf to illegal entries (here: large negative)
    masked_logits = np.where(mask, logits, -1e9)

    # 5) Optional temperature scaling
    if temperature is not None and temperature > 0:
        masked_logits = masked_logits / float(temperature)

    # 6) Softmax to probabilities (illegal ones become ~0, legal re-normalize to 1)
    probs = _softmax_np(masked_logits)

    # 7) Choose a legal move
    if sample:
        mv = features.sample_legal_move_from_probs(probs, board)
    else:
        # argmax among legal moves; uses masked logits for tie-breaking in logit space
        mv = features.best_legal_move_from_logits(masked_logits, board)

    return (mv, probs) if return_probs else (mv, None)
