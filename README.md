# Crazy House

A chess variant that adds shogi-style piece drops — captured pieces return to the capturer's hand and can be placed back on the board. The AI is powered by a Gumbel AlphaZero implementation (ResNet + Sequential Halving MCTS).

---

## Quick Start

### Run the EXE

1. Download `CrazyHouse.exe` from [Releases](https://github.com/mintBomber/CrazyHouse/releases) (or build it yourself — see [Building EXE](#9-building-exe))
2. Double-click `CrazyHouse.exe` — no installation required
3. A pre-trained model is bundled; you can play against the AI immediately

> The window is **resizable** by dragging the corner. Content scales automatically.

### Menu Buttons

| Button | Description |
|--------|-------------|
| **Player vs Player** | Two humans on the same machine |
| **Player vs AI** | Human vs the bundled AI |
| **AI vs AI** | Watch the AI play itself (batch mode available) |
| **Replay** | Load and replay a saved game record |
| **Model Learning** | Run additional self-play training from the UI |
| **Quit** | Exit the application |

### In-Game Controls

| Action | Input |
|--------|-------|
| Select a piece | Left-click your own piece on the board |
| Move / drop | Left-click a highlighted destination square |
| Select a hand piece (Crazy House) | Left-click a piece in the right-side hand panel |
| Deselect | Right-click anywhere |
| Open pause menu | Press **ESC** (or click the **[ESC] menu** button) |

### Left Panel at a Glance

| Label | Meaning |
|-------|---------|
| `Turn: White / Black` | Current player |
| `Move: N` | Half-move count (each individual move counts as 1) |
| `Training: N` | Number of self-play iterations the loaded model was trained for |
| Win gauge | Black% (left) · White% (right) — material-blended NN evaluation |
| `Best: …` / next candidates | Top policy-head candidate moves |
| `Main MM:SS  Move N.Ns` | Remaining main time and per-move byoyomi |

---

## Developer Documentation

### Table of Contents

1. [Setup (Python)](#1-setup)
2. [Running from Source](#2-running-from-source)
3. [Game Screens & Controls (detail)](#3-game-screens--controls-detail)
4. [Rules](#4-rules)
5. [Game Records](#5-game-records)
6. [AI Training](#6-ai-training)
7. [Configuration Reference](#7-configuration-reference)
8. [Architecture & File Structure](#8-architecture--file-structure)
9. [Building EXE](#9-building-exe)
10. [Tests](#10-tests)

---

## 1. Setup

**Requirements**

- Python 3.9+
- CPU-only is supported; CUDA-capable GPU strongly recommended for serious training

**Install**

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| `pygame` | GUI |
| `numpy` | Board representation and move generation |
| `torch` | Neural network (policy + value heads) |
| `pyyaml` | `config/config.yaml` loading |

---

## 2. Running from Source

```bash
# Default — loads checkpoints/{run_name}/latest.pt from config.yaml
python main.py

# Override run name (loads checkpoints/my_run/latest.pt)
python main.py --run my_run

# Load a specific checkpoint file
python main.py --ckpt checkpoints/run_quick_cpu/iter_00010.pt

# Run self-play training only (no GUI)
python -m ai.train
```

---

## 3. Game Screens & Controls (detail)

### Pre-game Setup

| Setting | Details |
|---------|---------|
| `Variant` | **Crazy House** (drops enabled) or **Standard Chess** |
| `First move` | Which color moves first |
| `Human side` | Which side the human plays (Player vs AI) |
| `Main time (min)` | Per-player main clock, 0–60 min |
| `One-move time (sec)` | Per-move byoyomi after main time runs out, 0–600 s |
| `Matches (AI vs AI)` | Number of consecutive AI vs AI games, 1–1000 |

Click a number field to type a value directly; `+` / `−` buttons adjust by 1.

### In-Game Panel

**Left sidebar**

```
Turn: White
AI thinking…          ← shown while AI is computing
Move: 28
50-move: 0/100
Training: 20
49%         51%       ← Black / White win probability
[====|=====]          ← evaluation gauge
Best: Nf3
d4, c4, e5            ← next 3 candidate moves
White (First)
  Main 03:50  Move 10.0s
Black (Second)
  Main 04:49  Move 10.0s
[Resign]
[Save Record]
AI vs AI
[ESC] menu
```

**Right sidebar (Crazy House only)**

- `Black's hand` — captured pieces available for Black to drop
- `White's hand` — same for White
- The active side's panel is highlighted with an orange border
- Greyed-out pieces have a count of ×0

### End-of-Game Screen

Outcomes: `Check Mate`, `Stale Mate`, `Draw`, `Time Up`, `Resign`

| Button | Action |
|--------|--------|
| `Replay` | Rematch with the same settings |
| `Top` | Return to the main menu |
| `Save Record` | Save the game as a JSON file |

### Replay Screen

| Button | Action |
|--------|--------|
| `<<` | Jump to move 0 |
| `<` | Step back one move |
| `Go` | Start auto-play (1 move per second) |
| `\|\|` | Stop auto-play |
| `>` | Step forward one move |
| `>>` | Jump to the last move |
| **[ESC] Top** | Return to the main menu (also clickable) |

The replay panel shows the same evaluation gauge and candidate moves that were recorded at each position during the original game. For records without stored evaluations the gauge is computed live from material count.

### AI vs AI Batch Mode

Setting `Matches` > 1 runs that many games automatically. On completion a result screen shows:

- Total matches, First-player wins, Second-player wins, Draws
- Average turn count
- Rule variant

Results are saved as JSON under `aivai_results/`.

### Model Learning (GUI)

**Model Learning** in the main menu opens a training dialog:

| Setting | Range |
|---------|-------|
| `Iterations` | 1–9999 additional training iterations |
| `Rule` | Crazy House / Standard |

Progress is shown as `Now: N / M steps`. On completion the AI agent is reloaded automatically.

---

## 4. Rules

### Standard Chess Rules (all apply)

| Rule | Status |
|------|--------|
| Castling | ✓ |
| En passant | ✓ |
| Pawn promotion | ✓ (Queen / Rook / Bishop / Knight) |
| Stalemate → draw | ✓ |
| 50-move rule (100 half-moves) | ✓ |
| Threefold repetition | ✓ |
| King capture | ✗ — game ends by checkmate only |

### Crazy House Additions

| Rule | Detail |
|------|--------|
| Captured pieces go to hand | Captured piece is demoted to its base type (no promoted pieces in hand) |
| Drop on empty square | Any hand piece can be placed on any empty square |
| No pawn drop on back rank | White pawns cannot be dropped on rank 8; Black pawns on rank 1 |
| No pawn-drop checkmate | Dropping a pawn that immediately checkmates is illegal (打ち歩詰め) |
| Non-pawn drop checkmate | Dropping any other piece that gives checkmate **is** legal |

---

## 5. Game Records

### Format

Records are saved as JSON under `gamerecord/`.  
File name pattern: `YYYYMMDD_HHMMSS_<save_name>.json`

Key fields in the record:

```json
{
  "game": "Chess x Shogi",
  "variant": "crazy_house",
  "move_count": 72,
  "result": "white_win",
  "moves": [
    {
      "move_number": 1,
      "player": "white",
      "notation": "e2e4",
      "check": false,
      "eval_winrate": 51.2,
      "eval_best": "e2e4",
      "eval_candidates": ["d2d4", "g1f3"]
    }
  ]
}
```

`eval_winrate`, `eval_best`, and `eval_candidates` are recorded for each move at the moment it is played (evaluation of the position *before* the move). These are replayed in the Replay screen without recomputation.

### Replay List

The Replay screen lists all files in `gamerecord/` sorted by modification time.

| Control | Action |
|---------|--------|
| Click a row | Open that record |
| Checkbox | Select for batch delete |
| `Delete` | Delete one record (with confirmation) |
| `Delete Selected` | Delete selected records |
| `Delete All` | Delete all records |

---

## 6. AI Training

### Algorithm

Gumbel AlphaZero ([Danihelka et al., 2022](https://arxiv.org/abs/2205.11093)):

1. Self-play using Gumbel MCTS (Sequential Halving with Gumbel noise at root)
2. Collect `(state, improved_policy, outcome)` tuples into a replay buffer
3. Train the network on mini-batches (policy cross-entropy + value MSE)
4. Checkpoint periodically

### Checkpoints

```
checkpoints/{run_name}/
├── iter_00010.pt
├── iter_00020.pt
└── latest.pt          ← always the most recent
```

### Current (lightweight) Config

The bundled `config.yaml` is tuned for CPU use:

```yaml
ai:
  device: "cpu"
  mcts_simulations: 25
network:
  num_res_blocks: 3
  channels: 64
training:
  run_name: "run_quick_cpu"
  total_iterations: 20
  num_self_play_games: 10   # → 200 total games
```

### Recommended Config for Stronger Play

```yaml
ai:
  device: "cuda"
  mcts_simulations: 400
network:
  num_res_blocks: 10
  channels: 256
  value_fc_size: 256
training:
  run_name: "run_strong"
  total_iterations: 1000
  num_self_play_games: 100  # → 100,000 total games
  batch_size: 256
  buffer_size: 100000
```

### Evaluation

The win-probability gauge blends the NN value head with a material-count heuristic:

```
nn_weight   = clamp(|nn_value| / 0.15,  0, 1)
win%        = nn_weight × nn_win% + (1 − nn_weight) × material_win%
```

An undertrained network outputs values near 0, so the material heuristic dominates until the network converges. Piece values: P=1, N=3, B=3, R=5, Q=9.

---

## 7. Configuration Reference

`config/config.yaml`

### `game`

| Key | Description |
|-----|-------------|
| `max_moves` | Max half-moves before draw |
| `board_size` | Always 8 |

### `ai`

| Key | Description |
|-----|-------------|
| `device` | `cpu` or `cuda` |
| `mcts_simulations` | Simulations per move |
| `gumbel_K` | Initial candidates for Gumbel Sequential Halving |
| `c_puct` | PUCT exploration constant (non-root nodes) |
| `dirichlet_alpha` | Root noise alpha (self-play only) |
| `dirichlet_eps` | Root noise mix ratio |
| `temperature` | Move sampling temperature |
| `temperature_threshold` | Move number after which temperature → 0 |

### `network`

| Key | Description |
|-----|-------------|
| `num_res_blocks` | Number of residual blocks |
| `channels` | Feature channels |
| `policy_channels` | Policy head intermediate channels |
| `value_channels` | Value head intermediate channels |
| `value_fc_size` | Value head FC hidden size |
| `input_planes` | Input feature planes (28 fixed) |

### `training`

| Key | Description |
|-----|-------------|
| `run_name` | Subdirectory name under `checkpoints/` |
| `total_iterations` | Total training iterations for this run |
| `checkpoint_dir` | Root checkpoint directory |
| `save_every_n_iters` | Checkpoint save frequency |
| `keep_last_n_checkpoints` | How many `iter_*.pt` files to keep (0 = all) |
| `resume` | Resume from `latest.pt` if present |
| `batch_size` | Training batch size |
| `lr` | Learning rate |
| `weight_decay` | L2 regularization |
| `num_self_play_games` | Self-play games per iteration |
| `num_epochs` | Training epochs per iteration |
| `buffer_size` | Replay buffer capacity |
| `drop_mode` | Enable Crazy House rules during self-play |
| `save_self_play_records` | Save self-play game JSONs |
| `self_play_record_dir` | Directory for self-play records |

### `ui`

| Key | Description |
|-----|-------------|
| `window_width`, `window_height` | Initial OS window size (logical canvas is fixed 1200×750) |
| `board_offset_x/y` | Board image top-left in logical canvas |
| `board_display_width` | Rendered board width |
| `board_col_starts`, `board_row_starts` | Grid pixel boundaries (auto-derived) |
| `fps` | Frame rate |
| `hand_panel_x` | Right panel X coordinate |
| `font_size` | Base font size |
| `colors` | UI colour overrides |

---

## 8. Architecture & File Structure

```
CrazyHouse/
├── main.py                  # CLI entry point
├── requirements.txt
├── CrazyHouse.spec          # PyInstaller spec
├── chess_board.png          # Board image (1602×1202)
├── chess_pieces.png         # Sprite sheet (2 rows × 6 cols)
├── config/
│   └── config.yaml
├── game/
│   ├── pieces.py            # PieceType, Color, Move, action-index mapping
│   └── board.py             # GameState: move gen, check, drop rules, terminal
├── ai/
│   ├── network.py           # ResNet: 28-plane input → policy (4416) + value
│   ├── mcts.py              # Gumbel MCTS with Sequential Halving
│   ├── agent.py             # AIAgent: wraps network + MCTS for the GUI
│   └── train.py             # Self-play training loop
├── ui/
│   ├── assets.py            # Image loading with alpha-channel sprite detection
│   └── ui.py                # Pygame UI (game loop, dialogs, replay, training UI)
├── tests/
│   └── test_hands_and_drops.py
├── checkpoints/
│   └── run_quick_cpu/
│       └── latest.pt        # Bundled pre-trained model
├── gamerecord/              # User game records (gitignored)
├── selfplay_records/        # Self-play training records (gitignored)
└── aivai_results/           # AI vs AI batch results (gitignored)
```

### Action Space

```
4416 total actions
├── 4096 board moves  (64 from-squares × 64 to-squares)
└──  320 drop moves   (5 piece types × 64 squares)
```

### Input Planes (28 channels, 8×8)

| Channels | Content |
|----------|---------|
| 0–5 | Current player's pieces (P R N B Q K) |
| 6–11 | Opponent's pieces |
| 12–16 | Current player's hand counts |
| 17–21 | Opponent's hand counts |
| 22–25 | Castling rights |
| 26 | En-passant target square |
| 27 | Side-to-move flag |

---

## 9. Building EXE

```bash
pip install pyinstaller
pyinstaller CrazyHouse.spec --clean
# Output: dist/CrazyHouse/CrazyHouse.exe
```

Notes:

- PyTorch makes the bundle large (~300 MB+); use `--onedir` (default) for faster startup
- For distribution, use the CPU-only PyTorch wheel to reduce size:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```
- Set `console=False` in the spec for release builds; `True` for debugging

---

## 10. Tests

```bash
python -m unittest discover -s tests
```

Current test coverage (`tests/test_hands_and_drops.py`):

- Captured piece enters the capturer's hand
- White/Black hand pieces are placed as the correct color
- Check-giving drops are legal
- Pawn-drop checkmate is illegal (打ち歩詰め)
- Non-pawn drop checkmate is legal
