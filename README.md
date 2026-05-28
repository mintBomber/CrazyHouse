# Chess × Shogi — Gumbel AlphaZero

チェスをベースに将棋の持ち駒ルールを導入したハイブリッドボードゲームです。
AIには **Gumbel AlphaZero**（Sequential Halving + ResNet）を採用しています。

---

## 目次

1. [セットアップ](#1-セットアップ)
2. [実行方法](#2-実行方法)
3. [ゲームルール](#3-ゲームルール)
4. [操作方法](#4-操作方法)
5. [AIアーキテクチャ](#5-aiアーキテクチャ)
6. [ファイル構成](#6-ファイル構成)
7. [設定ファイル詳解](#7-設定ファイル詳解)
8. [AIの学習方法](#8-aiの学習方法)

---

## 1. セットアップ

### 必要環境
- Python 3.9 以上
- GPU（オプション、CPUでも動作します）

### インストール

```bash
cd Game02
pip install -r requirements.txt
```

`requirements.txt` の内容：

| パッケージ | 用途 |
|-----------|------|
| `pygame`  | ゲームUI |
| `numpy`   | 盤面の数値演算 |
| `torch`   | ニューラルネットワーク |
| `pyyaml`  | 設定ファイルの読み込み |

---

## 2. 実行方法

### GUIでゲームを起動

```bash
python main.py
```

起動するとメニュー画面が表示されます。3つのモードから選択してください。

### 学習済みモデルを指定して起動

```bash
python main.py --ckpt checkpoints/run_001/latest.pt
```

### 学習 run を指定して起動

```bash
python main.py --run run_001
```

### 自己対戦学習のみ実行（GUI なし）

```bash
python -m ai.train
```

### モード選択画面

| モード | 説明 |
|--------|------|
| **Player vs Player** | 2人のプレイヤーが同じ画面で対戦 |
| **Player vs AI**     | あなた（白）対AI（黒） |
| **AI vs AI**         | AIどうしが自動対戦（観戦モード） |

### 棋譜を保存する

対局中、左側の情報パネルにある **Save GameRecord** ボタンを押すと、その時点までの棋譜を保存できます。終局後の画面にも同じボタンがあり、終わった対局の棋譜を保存できます。

保存先：

```text
gamerecord\
```

保存ファイル名の例：

```text
gamerecord\20260529_153012_pvai_042moves.json
```

棋譜ファイルには、対局モード、開始時刻、保存時刻、結果、各手の `notation`、移動元・移動先、持ち駒打ち、成り情報、最終盤面が JSON 形式で保存されます。

---

## 3. ゲームルール

### 基本ルール（標準チェスと同じ）

| ルール | 内容 |
|--------|------|
| 目的 | 相手のキングをチェックメイト（詰み）にする |
| 駒の動き | 標準チェスと完全に同一 |
| キャスリング | キングサイド・クイーンサイドともに有効 |
| アンパッサン | 有効（ポーンが2マス前進した直後のみ取れる） |
| ポーンの成り | 最終段に達したとき、クイーン・ルーク・ビショップ・ナイトに成れる |
| 引き分け（50手ルール） | 駒の取り合いもポーンの前進もなく50手が経過したら引き分け |
| 引き分け（千日手） | 同一局面が3回現れたら引き分け |
| 引き分け（ステイルメイト） | 手番の側が合法手を持たず、チェックでもない状態 |

### 追加ルール（将棋方式の持ち駒）

#### 持ち駒と打ち込み

取った相手の駒は **自分の持ち駒** になります。  
自分の手番に、持ち駒を **空きマスに打ち込む** ことができます。

```
例：白がポーンを取った場合
  → 白の手駒に「ポーン ×1」が追加される
  → 次以降の自分の手番に、好きな空きマスに打てる
```

打ち込んだ駒はそのまま通常の駒として動かせます。  
ポーンを打った後に最終段に到達すれば、通常通り成れます。

#### 打ち禁ルール

以下の場合は **打てません**：

| 禁止ルール | 内容 |
|-----------|------|
| **ポーンの最終段打ち禁止** | 白のポーンを8段目（最終段）に打つことはできない。黒のポーンを1段目に打つことも不可 |
| **打ち歩詰め禁止（将棋と同ルール）** | ポーンを打って、その手番で相手をチェックメイトにすることはできない。チェックにすること自体は合法 |

> **注意：** キングは永遠に取られないため持ち駒には加わりません。

---

## 4. 操作方法

### マウス操作

| 操作 | 内容 |
|------|------|
| **盤面の自分の駒を左クリック** | 駒を選択（緑でハイライト）。合法な移動先が青いドットで表示される |
| **青いドット（移動先）を左クリック** | 選択した駒を移動。相手の駒がいる場合はリング表示 |
| **手駒パネルの駒を左クリック** | 打つ駒を選択（オレンジでハイライト）。打てるマスが青いドットで表示される |
| **青いドット（打ち先）を左クリック** | 選択した手駒を打ち込む |
| **右クリック / 他のマスをクリック** | 選択を解除 |
| **ESCキー** | メインメニューに戻る |

### 画面レイアウト

```
┌──────────────┬──────────────────────────┬────────────────┐
│              │                          │  Black's hand  │
│  情報パネル   │       チェス盤            │   (上側)       │
│              │  (chess_board.png)        │                │
│  ・手番表示   │   駒表示                  │────────────────│
│  ・手数       │   ハイライト              │                │
│  ・チェック表示│                          │  White's hand  │
│  ・モード表示  │                          │   (下側)       │
│  ・[ESC]メニュー│                         │                │
└──────────────┴──────────────────────────┴────────────────┘
```

### 成り（プロモーション）ダイアログ

ポーンが最終段に到達すると、自動的にダイアログが表示されます。

```
  [ Q ]  [ R ]  [ B ]  [ N ]
クイーン  ルーク  ビショップ  ナイト
```

クリックして成り先の駒を選択してください。

### 強調表示の色

| 色 | 意味 |
|----|------|
| 緑 | 選択中の駒 |
| 青（ドット） | 移動・打ち先の候補 |
| 黄 | 直前の手（移動元・移動先） |
| 赤 | チェックされているキング |
| オレンジ | 選択中の手駒 |

---

## 5. AIアーキテクチャ

### Gumbel AlphaZero 概要

Google DeepMindの論文 *"Policy improvement by planning with Gumbel"*（Danihelka et al., ICLR 2022）に基づく実装です。標準のAlphaZeroと比べて **少ないシミュレーション数で強い手を選べる** のが特徴です。

### ニューラルネットワーク

**ファイル：** `ai/network.py`

```
入力 (1, 28, 8, 8)
  ↓
Conv 3×3 → BatchNorm → ReLU  [28 → 256ch]
  ↓
ResBlock × 10  [256ch]
  ↙                 ↘
[Policy Head]      [Value Head]
Conv 1×1 (2ch)     Conv 1×1 (1ch)
Flatten             Flatten
FC → 4416          FC(256) → FC(1) → tanh
(手の確率分布)      (-1〜+1 の評価値)
```

#### 入力の28チャンネル

| チャンネル | 内容 |
|-----------|------|
| 0〜5 | 現在の手番側の駒の位置（P R N B Q K 各1枚） |
| 6〜11 | 相手側の駒の位置（P R N B Q K 各1枚） |
| 12〜16 | 現在の手番側の持ち駒枚数（P R N B Q）※最大枚数で正規化 |
| 17〜21 | 相手側の持ち駒枚数（P R N B Q） |
| 22〜25 | キャスリング権（白K・白Q・黒K・黒Q） |
| 26 | アンパッサンの対象マス |
| 27 | 手番（白=1.0 全面、黒=0.0 全面） |

> 盤面は常に **現在の手番側から見た向き**（自分の陣地が下）に正規化されるため、白番・黒番で同じネットワークが使えます。

#### アクション空間（4416）

| 範囲 | 内容 |
|------|------|
| 0〜4095 | 盤上の指し手（from_sq × 64 + to_sq） |
| 4096〜4415 | 打ち駒（piece_idx × 64 + to_sq）piece_idx: P=0 R=1 N=2 B=3 Q=4 |

### Gumbel Sequential Halving（MCTS）

**ファイル：** `ai/mcts.py`

```
1. ルートノードをネットワークで評価 → 方策 π と評価値 v を取得

2. 各合法手 a にGumbelノイズを加えてスコアを計算
   score(a) = g_a + log π(a)    (g_a ~ Gumbel(0,1))

3. 上位K手を候補として選択（config: gumbel_K）

4. Sequential Halving（候補をlog₂(K)フェーズで半分ずつ絞る）
   各フェーズ：
     ・各候補について MCTS シミュレーションを実行
     ・Q値（平均バックアップ値）が低い下位半数を除外

5. 残った候補の訪問回数から改良方策 π_improved を計算して返す
```

**ルート以外のノード選択：** 標準の PUCT
```
UCB(a) = Q(a) + c_puct × π(a) × √N / (1 + n(a))
```

### エージェント

**ファイル：** `ai/agent.py`

- `AIAgent.select_move(state, move_number)` — 指し手を1手返す
- 序盤（`move_number < temperature_threshold`）は確率的サンプリング、終盤は最大訪問回数の手（greedy）を選択
- `save_checkpoint()` / `load_state_dict()` でモデルの保存・復元

---

## 6. ファイル構成

```
Game02/
│
├── main.py                  # エントリポイント
├── requirements.txt         # 依存パッケージ
│
├── config/
│   └── config.yaml          # 全設定パラメータ
│
├── game/                    # ゲームエンジン（AIとUIに依存しない）
│   ├── __init__.py
│   ├── pieces.py            # PieceType・Color・Move・アクションエンコード
│   └── board.py             # GameState（盤面・合法手生成・詰み判定）
│
├── ai/                      # AI（ゲームエンジンのみに依存）
│   ├── __init__.py
│   ├── network.py           # ResNet ポリシー・バリューネットワーク
│   ├── mcts.py              # Gumbel AlphaZero MCTS
│   ├── agent.py             # AIAgent（MCTS + チェックポイント管理）
│   └── train.py             # 自己対戦学習ループ
│
└── ui/                      # Pygame UI（ゲームエンジンとAIに依存）
    ├── __init__.py
    ├── assets.py            # chess_pieces.png の分割・読み込み
    └── ui.py                # メニュー・ゲームループ・描画・入力処理
```

### 依存関係

```
game/pieces.py
    └─ game/board.py
         ├─ ai/network.py
         │    ├─ ai/mcts.py
         │    │    └─ ai/agent.py
         │    └─ ai/train.py
         └─ ui/assets.py
              └─ ui/ui.py ──── ai/agent.py
                               main.py
```

---

## 7. 設定ファイル詳解

**ファイル：** `config/config.yaml`

### `game` セクション

```yaml
game:
  max_moves: 500     # 1ゲームの最大半手数（これを超えたら引き分け）
  board_size: 8      # 盤面サイズ（変更不要）
```

### `ai` セクション（対局時のAI設定）

```yaml
ai:
  device: "cpu"              # "cuda" にするとGPU使用（要PyTorch CUDA版）
  mcts_simulations: 400      # 1手あたりのシミュレーション数（多いほど強い・遅い）
  gumbel_K: 16               # Gumbel選択の初期候補数（2の累乗推奨）
  c_puct: 1.25               # PUCT探索の定数（大きいほど探索重視）
  dirichlet_alpha: 0.3       # 学習時のルートノードへのディリクレノイズ強度
  dirichlet_eps: 0.25        # ノイズの混合割合（0=ノイズなし, 1=ノイズのみ）
  temperature: 1.0           # 指し手サンプリング温度（0=greedy）
  temperature_threshold: 30  # この手数以降はgreedy選択（温度0）
  value_scale: 1.0           # バリューヘッドの出力スケール（通常変更不要）
```

**AIの強さを調整するには：**

| 目的 | 変更する項目 |
|------|------------|
| AIを強くする | `mcts_simulations` を増やす（例：800〜1600） |
| AIを速くする | `mcts_simulations` を減らす（例：100〜200） |
| GPU使用 | `device: "cuda"` に変更 |
| 探索の幅を広げる | `gumbel_K` を増やす（例：32） |

### `network` セクション（ネットワーク構造）

```yaml
network:
  num_res_blocks: 10     # 残差ブロック数（多いほど強い・重い）
  channels: 256          # 各層のチャンネル数
  policy_channels: 2     # ポリシーヘッドの中間チャンネル数
  value_channels: 1      # バリューヘッドの中間チャンネル数
  value_fc_size: 256     # バリューヘッドの全結合層サイズ
  input_planes: 28       # 入力チャンネル数（変更不要）
```

**ネットワークを小さくしたい場合（低スペックPC向け）：**

```yaml
network:
  num_res_blocks: 5
  channels: 128
  value_fc_size: 128
```

### `training` セクション（学習設定）

```yaml
training:
  run_name: "run_001"           # 学習セッション名
  total_iterations: 1000        # この run で実行する総イテレーション数
  checkpoint_dir: "checkpoints" # チェックポイントのルートディレクトリ
  save_every_n_iters: 10        # N イテレーションごとに保存
  keep_last_n_checkpoints: 0    # 0=全保持、1以上=直近N個だけ保持
  resume: true                  # latest.pt があれば再開
  batch_size: 256              # ミニバッチサイズ
  lr: 0.001                    # 初期学習率
  weight_decay: 0.0001         # L2正則化
  num_self_play_games: 100     # 1イテレーション当たりの自己対戦ゲーム数
  num_epochs: 10               # 1イテレーション当たりの学習エポック数
  buffer_size: 100000          # リプレイバッファの最大サイズ
```

### `ui` セクション（UI表示設定）

```yaml
ui:
  window_width: 1200            # ウィンドウ幅（ピクセル）
  window_height: 800            # ウィンドウ高さ
  board_offset_x: 150          # 盤面画像の左端X座標
  board_offset_y: 50           # 盤面画像の上端Y座標
  board_display_width: 890     # 盤面画像の表示幅（高さはアスペクト比から自動計算）
  # 盤面画像内の8x8グリッド開始位置（chess_board.png のピクセル解析から算出）
  board_inner_x: 190           # グリッド左端までのオフセット（px）
  board_inner_y: 76            # グリッド上端までのオフセット（px）
  board_sq_size: 64            # 各マス目のピクセルサイズ
  fps: 60                      # フレームレート
  animation_speed: 150         # AI vs AI での1手あたりの待機時間（ミリ秒）
  hand_panel_x: 1060           # 手駒パネルのX座標
  hand_piece_size: 52          # 手駒アイコンのサイズ（ピクセル）
```

---

## 8. AIの学習方法

### 学習の仕組み

```
[自己対戦]
  AIが自分自身と対戦して棋譜データを生成
  各局面で (状態, 改良方策, 結果) を記録

        ↓

[学習]
  ネットワークを棋譜データで最適化
  損失関数 = 方策のクロスエントロピー + 評価値のMSE

        ↓

[チェックポイント保存]
  checkpoints/{run_name}/iter_XXXXX.pt
  checkpoints/{run_name}/latest.pt  ← GUIが自動的にこれを読む
```

### 学習を始める

```bash
python -m ai.train
```

### 10000局学習させる具体例

この実装では、自己対戦の総局数は次の式で決まります。

```text
total_iterations × num_self_play_games = 自己対戦の総局数
```

10000局にしたい場合は、例えば `100 iteration × 100局 = 10000局` にします。
[config/config.yaml](config/config.yaml) の `training` を次のように設定します。

```yaml
training:
  run_name: "run_10000_games"
  total_iterations: 100
  checkpoint_dir: "checkpoints"
  save_every_n_iters: 10
  keep_last_n_checkpoints: 10
  resume: true

  batch_size: 256
  lr: 0.001
  lr_decay_steps: [100000, 300000]
  lr_decay_gamma: 0.1
  weight_decay: 0.0001
  num_self_play_games: 100
  num_epochs: 10
  buffer_size: 100000
```

この設定で学習を開始します。

```bash
cd D:\Programming\PersonalDevelopments\Game02
python -m ai.train
```

途中で止めた場合も、`resume: true` かつ同じ `run_name` であれば、同じコマンドで続きから再開します。

### ノード展開数に相当する設定

1手あたりの探索量は `ai.mcts_simulations` で指定します。厳密には「展開ノード数」という名前ではなく、MCTS のシミュレーション回数です。

```yaml
ai:
  mcts_simulations: 100
```

値を大きくすると1手ごとの読みが深くなりやすい一方で、自己対戦学習も対局時のAI思考も遅くなります。目安は次の通りです。

| 目的 | 設定例 |
|------|--------|
| 動作確認を速く済ませる | `mcts_simulations: 50` |
| CPUで10000局を現実的に回す | `mcts_simulations: 100` |
| 対局時の読みを強くする | `mcts_simulations: 400` |

学習時は `100`、学習後にAIと対戦するときだけ `400` に戻す、という使い方もできます。モデルの重みは変わらず、起動後の探索量だけが変わります。

バックグラウンドで長時間実行する場合（Windows）：

```bash
start /B python -m ai.train > train_log.txt 2>&1
```

### 学習の進捗確認

ターミナルに以下の形式で出力されます：

```
10 / 10000 steps
20 / 10000 steps
...
[iter 1]  policy_loss=8.3921  value_loss=0.9987  time=142.3s
[iter 10]  policy_loss=7.1204  value_loss=0.8843  time=138.7s
  [iter 10] Checkpoint saved: checkpoints/run_001/iter_00010.pt
```

ここでの `steps` は自己対戦の局数です。例えば `total_iterations: 100`、`num_self_play_games: 100` なら合計 `10000 steps` になります。

- `policy_loss` が下がる → AIの指し手選択が棋譜に近づいている
- `value_loss` が下がる → 勝敗予測が正確になっている

### チェックポイントの管理

| ファイル | 内容 |
|---------|------|
| `checkpoints/run_001/latest.pt` | run_001 の最新モデル（GUIが自動読み込み） |
| `checkpoints/run_001/iter_00010.pt` | run_001 のイテレーション10時点のスナップショット |

`run_name: "run_10000_games"` で学習した場合、モデルは次の場所に保存されます。

```text
D:\Programming\PersonalDevelopments\Game02\checkpoints\run_10000_games\
```

代表的なファイル名は次の通りです。

```text
checkpoints\run_10000_games\iter_00010.pt
checkpoints\run_10000_games\iter_00020.pt
...
checkpoints\run_10000_games\iter_00100.pt
checkpoints\run_10000_games\latest.pt
```

`iter_XXXXX.pt` はそのイテレーション時点のスナップショットです。`latest.pt` は常に最新モデルで上書きされ、通常の対戦ではこのファイルを使います。

特定のチェックポイントでGUIを起動：

```bash
python main.py --ckpt checkpoints/run_001/iter_00050.pt
```

10000局学習した最新モデルでAIと対戦する場合：

```bash
python main.py --run run_10000_games
```

または、チェックポイントを直接指定します。

```bash
python main.py --ckpt checkpoints\run_10000_games\latest.pt
```

### 学習を再開する

`training.resume: true` の場合、同じ `run_name` で再実行すると `checkpoints/{run_name}/latest.pt` から自動的に再開します。

```bash
python -m ai.train   # 中断した続きから再開
```

### 学習パラメータのチューニング指針

| 状況 | 推奨設定 |
|------|---------|
| まず動かして試したい | `num_self_play_games: 10`, `mcts_simulations: 50` |
| 本格的に学習させたい | デフォルト設定、GPU使用（`device: "cuda"`） |
| メモリが不足する | `buffer_size: 50000`, `batch_size: 128` |
| 学習が発散する | `lr: 0.0003` に下げる |

---

## ライセンス / 備考

- AIは初期状態ではランダム初期化なので、学習前は弱いです。`python -m ai.train` で学習を重ねるほど強くなります。
- 学習なしでも Player vs Player で遊べます。
- 画像ファイル `chess_pieces.png` と `chess_board.png` はゲームルートディレクトリに必要です。
