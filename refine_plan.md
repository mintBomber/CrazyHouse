# 改善計画：学習・推論の分離と学習管理の強化

作成日：2026-05-29  
対象リポジトリ：`D:\Programming\PersonalDevelopments\Game02`

---

## 1. 現状の問題点

### 1-1. 毎回モデルを生成している

`ai/agent.py` の `AIAgent.__init__` は、チェックポイントが無ければ  
ランダム初期化済みの `AlphaZeroNet` を生成し、そのまま対局に使う。

`ui/ui.py` の `_start_game()` が `AIAgent(cfg, ckpt_path)` を呼ぶため、  
**ゲームを起動するたびに毎回ネットワークオブジェクトが生成・破棄される。**

```
# ui/ui.py _start_game() 現状
if mode == "pvai":
    self.ai_white = None
    self.ai_black = AIAgent(self.cfg, ckpt_path)   ← 毎回生成
```

### 1-2. 学習コマンドと起動コマンドが混在している

`main.py --train` で学習が走るが、学習回数の指定やモデル保存名の制御が  
コードとYAMLの間で曖昧になっている。

### 1-3. チェックポイントの命名が粗い

```
checkpoints/latest.pt         ← 上書きされ続ける
checkpoints/iter_00010.pt     ← イテレーション番号だが連番管理のみ
```

学習セッションごとの識別子（名前・日時）がないため、どの実験か追えない。

---

## 2. 実装方針

### 方針 A：学習は `train.py` 単体で完結させる

- `python train.py` または `python -m ai.train` だけで学習が動く。
- `main.py` は **UI起動専用** に限定し、学習ロジックを一切持たない。
- `main.py --train` フラグは削除する。

### 方針 B：学習パラメータはすべて `config/config.yaml` で管理する

追加すべきキー（後述）：

```yaml
training:
  run_name: "run_001"          # 学習セッションの識別名（必須）
  total_iterations: 500        # この run で行うイテレーション数
  save_every_n_iters: 10       # N イテレーションごとに保存
  keep_last_n_checkpoints: 5   # 古いチェックポイントを自動削除する個数（0=全保持）
  resume: true                 # true = latest.pt から自動再開、false = ゼロから開始
```

### 方針 C：チェックポイントのディレクトリ構造を整理する

```
checkpoints/
  run_001/
    iter_00010.pt
    iter_00020.pt
    ...
    iter_00500.pt   ← total_iterations 到達
    latest.pt       ← 常に最新を指すシンボリックコピー or 実体
  run_002/
    ...
  best.pt           ← (将来) 評価対局で選ばれた最良モデル
```

- 保存ファイル名：`{run_name}/iter_{iter:05d}.pt`
- 常に `{run_name}/latest.pt` を最新チェックポイントで上書きする。
- UI のデフォルトロードパスは `checkpoints/{run_name}/latest.pt`。

---

## 3. 変更が必要なファイルの一覧

| ファイル | 変更種別 | 概要 |
|---------|---------|------|
| `config/config.yaml` | 追記 | `training.run_name`, `total_iterations`, `save_every_n_iters`, `keep_last_n_checkpoints`, `resume` を追加 |
| `ai/train.py` | 書き直し | `run_name` ベースのチェックポイントディレクトリ管理、`total_iterations` 終了判定、古いチェックポイント自動削除 |
| `ai/agent.py` | 修正 | チェックポイントの探索ロジックを `run_name` ディレクトリに対応 |
| `main.py` | 修正 | `--train` フラグを削除。`--run` で `run_name` を上書き指定できるように変更 |
| `ui/ui.py` | 修正 | AIAgent 生成を `_start_game()` の度ではなく、ゲーム初回起動時に一度だけ行う（後述） |

---

## 4. 各ファイルの変更詳細

### 4-1. `config/config.yaml` — `training` セクションの追記

```yaml
training:
  run_name: "run_001"           # 学習識別名。チェックポイントは checkpoints/{run_name}/ に保存される
  total_iterations: 500         # この run_name で実行するイテレーション総数
  checkpoint_dir: "checkpoints" # ルートディレクトリ（変更不要）
  save_every_n_iters: 10        # N イテレーションごとに iter_XXXXX.pt を保存
  keep_last_n_checkpoints: 0    # 古いチェックポイントを何世代保持するか（0=全保持）
  resume: true                  # true=latest.pt から再開, false=ゼロ初期化から開始
  # 以下は既存パラメータ（変更なし）
  batch_size: 256
  lr: 0.001
  ...
```

### 4-2. `ai/train.py` — 主要ロジックの変更点

#### (a) チェックポイントパスの生成

```python
def ckpt_dir(cfg) -> str:
    """例: checkpoints/run_001"""
    return os.path.join(cfg['training']['checkpoint_dir'],
                        cfg['training']['run_name'])

def ckpt_path(cfg, iteration: int) -> str:
    """例: checkpoints/run_001/iter_00050.pt"""
    return os.path.join(ckpt_dir(cfg), f"iter_{iteration:05d}.pt")

def latest_path(cfg) -> str:
    """例: checkpoints/run_001/latest.pt"""
    return os.path.join(ckpt_dir(cfg), "latest.pt")
```

#### (b) 再開ロジック

```python
def load_or_init(cfg, device) -> tuple[AlphaZeroNet, Adam, int]:
    net = build_network(cfg['network'], device)
    opt = Adam(net.parameters(), lr=cfg['training']['lr'],
               weight_decay=cfg['training']['weight_decay'])
    start_iter = 0

    if cfg['training']['resume']:
        latest = latest_path(cfg)
        if os.path.isfile(latest):
            ckpt = torch.load(latest, map_location=device)
            net.load_state_dict(ckpt['model_state_dict'])
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            start_iter = ckpt['iteration'] + 1
            print(f"[train] Resumed from {latest} (iter={ckpt['iteration']})")
        else:
            print("[train] No checkpoint found; starting from scratch.")
    else:
        print("[train] resume=false; starting from scratch.")

    return net, opt, start_iter
```

#### (c) 終了条件

```python
total = cfg['training']['total_iterations']
for iteration in range(start_iter, total):
    # ... 自己対戦 + 学習 ...
    pass
print(f"[train] Completed {total} iterations for run '{cfg['training']['run_name']}'.")
```

#### (d) 保存ロジック

```python
def save(cfg, net, opt, iteration):
    os.makedirs(ckpt_dir(cfg), exist_ok=True)
    path = ckpt_path(cfg, iteration)
    data = {
        'model_state_dict':     net.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'iteration':            iteration,
        'run_name':             cfg['training']['run_name'],
    }
    torch.save(data, path)
    # 常に latest.pt を上書き
    torch.save(data, latest_path(cfg))
    print(f"[train] Saved: {path}")

    # 古いチェックポイントを削除
    keep = cfg['training'].get('keep_last_n_checkpoints', 0)
    if keep > 0:
        _prune_old_checkpoints(cfg, keep)

def _prune_old_checkpoints(cfg, keep: int):
    pattern = os.path.join(ckpt_dir(cfg), "iter_*.pt")
    import glob
    files = sorted(glob.glob(pattern))  # 辞書順 = イテレーション昇順
    for old in files[:-keep]:
        os.remove(old)
        print(f"[train] Removed old checkpoint: {old}")
```

### 4-3. `main.py` — `--train` 削除、`--run` 追加

```python
# 変更後の引数
parser.add_argument("--run", type=str, default=None,
                    help="run_name を上書き指定（config.yaml の値より優先）")
parser.add_argument("--ckpt", type=str, default=None,
                    help="UIで使うチェックポイントファイルを直接指定")
# --train は削除
```

学習は `python -m ai.train` または `python ai/train.py` で実行。  
`main.py` は **UI 専用エントリポイント** になる。

```python
# main.py
def main():
    cfg = _load_config()
    if args.run:
        cfg['training']['run_name'] = args.run

    # チェックポイントの解決順序
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        run_name = cfg['training']['run_name']
        ckpt_path = os.path.join(cfg['training']['checkpoint_dir'],
                                 run_name, 'latest.pt')

    from ui.ui import ChessUI
    ChessUI(cfg, asset_dir, default_ckpt=ckpt_path).run()
```

### 4-4. `ui/ui.py` — AIAgent の生成を一度だけにする

#### 現状の問題

`_start_game()` を呼ぶたびに `AIAgent(cfg, ckpt_path)` が実行される。  
AIvAI モードなら毎回 2 つのモデルが生成・破棄される。

#### 変更方針

`AIAgent` のインスタンスを `ChessUI.__init__` で **一度だけ** 生成してキャッシュする。

```python
class ChessUI:
    def __init__(self, cfg, asset_dir, default_ckpt=None):
        ...
        # 起動時に一度だけ生成（重い処理）
        from ai.agent import AIAgent
        self._shared_agent = AIAgent(cfg, default_ckpt)

    def _start_game(self, mode):
        ...
        if mode == "pvai":
            self.ai_white = None
            self.ai_black = self._shared_agent   # ← キャッシュを使い回す
        elif mode == "aivai":
            self.ai_white = self._shared_agent
            self.ai_black = self._shared_agent   # 同一ネットワーク
```

> **注意：** AIvAI の場合は同一エージェントが双方に使われる。  
> 将来「2つの異なるモデルを対局させる」機能を追加するときは  
> `ai_white_ckpt` / `ai_black_ckpt` を config に追加する。

---

## 5. 実行フローの変更まとめ

### 学習フロー（変更後）

```
# ゼロから学習（run_001）
python -m ai.train
  → config.yaml の run_name="run_001", total_iterations=500 を読む
  → resume=false なら初期化
  → checkpoints/run_001/iter_00010.pt, iter_00020.pt, ... を保存
  → checkpoints/run_001/latest.pt を常に最新で上書き

# 学習を中断して再開
python -m ai.train
  → resume=true なので checkpoints/run_001/latest.pt を読んで続き

# 新しい実験を始める（run_002）
# config.yaml で run_name: "run_002" に変更してから
python -m ai.train
  → checkpoints/run_002/ に保存開始
```

### ゲーム起動フロー（変更後）

```
# デフォルト（config の run_name の latest.pt を使う）
python main.py

# 特定の run を指定
python main.py --run run_001

# 特定のチェックポイントを指定
python main.py --ckpt checkpoints/run_001/iter_00200.pt

# 学習なし（ランダムAI）
python main.py --ckpt ""   # or just python main.py if no latest.pt exists
```

---

## 6. 実装優先度

| 優先度 | タスク | 難易度 |
|--------|--------|--------|
| 高 | `config.yaml` に `run_name`, `total_iterations`, `resume` を追加 | 低 |
| 高 | `ai/train.py` のチェックポイント管理を `run_name` ベースに書き直し | 中 |
| 高 | `main.py` から `--train` を削除、`--run` を追加 | 低 |
| 中 | `ui/ui.py` の `AIAgent` 生成をキャッシュ化 | 低 |
| 中 | `keep_last_n_checkpoints` による自動削除 | 低 |
| 低 | `best.pt` の概念（評価対局によるモデル選択） | 高 |

---

## 7. 現状コードの依存関係メモ（実装者向け）

```
main.py
  ├─ ui/ui.py  ← _start_game() で AIAgent を生成している（要修正）
  │    └─ ai/agent.py  ← checkpoint_path を受け取る
  │         └─ ai/network.py
  │         └─ ai/mcts.py
  └─ (削除予定) --train → ai/train.py

ai/train.py  ← 独立した学習スクリプト（main.py から切り離す）
  └─ ai/network.py
  └─ ai/mcts.py
  └─ game/board.py
```

## 8. 注意事項・既知の制約

1. **PyTorch `weights_only` 警告**  
   `torch.load()` に `weights_only=True` を付けると optimizer state が読めない場合がある。  
   PyTorch 2.x 以降は `weights_only=False` を明示するか、pickle safe な形式に変換する。

2. **Windows パス区切り**  
   `os.path.join` を使えば問題なし。`glob.glob` も Windows 対応済み。

3. **シンボリックリンク**  
   Windows では `latest.pt` をシンボリックリンクにするには管理者権限が必要なため、  
   `torch.save()` で**実体ファイルをコピー**する方式にしている（計画通り）。

4. **学習中断時の再開精度**  
   リプレイバッファ（deque）は保存していない。  
   再開後は空のバッファから自己対戦を再開するため、最初の数イテレーションはバッファが少ない。  
   将来的には `pickle` でバッファも保存する拡張が可能。
