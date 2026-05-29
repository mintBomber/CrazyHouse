# Chess x Shogi - Gumbel AlphaZero

チェスをベースに、将棋の「取った駒を持ち駒として打つ」ルールを加えたハイブリッドボードゲームです。
GUI は Pygame、AI は Gumbel AlphaZero 形式の MCTS + ResNet で実装しています。

## 目次

1. [セットアップ](#1-セットアップ)
2. [起動方法](#2-起動方法)
3. [ゲーム画面と操作](#3-ゲーム画面と操作)
4. [ルール](#4-ルール)
5. [棋譜保存と再生](#5-棋譜保存と再生)
6. [AI 学習](#6-ai-学習)
7. [設定ファイル](#7-設定ファイル)
8. [ファイル構成](#8-ファイル構成)
9. [テスト](#9-テスト)

## 1. セットアップ

必要環境:

- Python 3.9 以上
- CPU でも動作可能
- GPU を使う場合は CUDA 対応版 PyTorch が必要

インストール:

```powershell
cd D:\Programming\PersonalDevelopments\Game02
pip install -r requirements.txt
```

依存パッケージ:

| パッケージ | 用途 |
| --- | --- |
| `pygame` | ゲーム UI |
| `numpy` | 盤面・合法手計算 |
| `torch` | ニューラルネットワーク |
| `pyyaml` | `config/config.yaml` の読み込み |

## 2. 起動方法

通常起動:

```powershell
python main.py
```

学習 run 名を指定して起動:

```powershell
python main.py --run run_quick_cpu
```

この場合、AI は次のモデルを探します。

```text
checkpoints\run_quick_cpu\latest.pt
```

チェックポイントファイルを直接指定:

```powershell
python main.py --ckpt checkpoints\run_quick_cpu\latest.pt
```

AI 学習のみ実行:

```powershell
python -m ai.train
```

## 3. ゲーム画面と操作

### トップ画面

トップ画面には以下のボタンがあります。

| ボタン | 内容 |
| --- | --- |
| `Player vs Player` | 人間同士で対戦 |
| `Player vs AI` | 人間と AI が対戦 |
| `AI vs AI` | AI 同士の自動対戦を観戦 |
| `Replay GameRecord` | 保存済み棋譜を再生 |
| `Quit` | アプリを終了 |

### 対局前設定

対局モードを選ぶと、開始前に以下を設定できます。

| 項目 | 内容 |
| --- | --- |
| `First move` | 白番・黒番のどちらを先手にするか |
| `Human side` | Player vs AI で人間が先手か後手か |
| `Player 1` | Player vs Player で Player 1 が先手か後手か |
| `Main time (min)` | 各プレイヤーの持ち時間。範囲は `0-60` 分 |
| `One-move time (sec)` | 持ち時間を使い切った後の 1 手秒読み。範囲は `0-600` 秒 |

`Main time` と `One-move time` は、`+` / `-` ボタンでも、数字欄の直接入力でも変更できます。
`One-move time` は 1 秒単位で変化します。

### 対局中の操作

| 操作 | 内容 |
| --- | --- |
| 盤上の自分の駒を左クリック | 駒を選択 |
| 青い候補マスを左クリック | 選択中の駒を移動 |
| 右サイドバーの自分の持ち駒を左クリック | 打つ持ち駒を選択 |
| 青い候補マスを左クリック | 選択中の持ち駒を盤上に打つ |
| 右クリック | 選択解除 |
| `ESC` | トップ画面へ戻る |

右サイドバーには `Black's hand` と `White's hand` が表示されます。
現在手番側の持ち駒欄は枠で強調されます。
持ち駒を打つと、その駒は自分の色の駒として盤上に置かれます。

### 左サイドバー

左サイドバーには以下が表示されます。

- 現在の手番
- 手数
- 50 手ルールカウンタ
- `Front win`、手前側の白番視点の勝率
- 白黒それぞれの残り持ち時間と 1 手秒読み
- `CHECK!` 表示
- `Resign` ボタン
- `Save Record` ボタン

`Resign` を押すと `Really Quit？` ダイアログが表示されます。
`Yes` を押すと現在手番が投了し、相手勝ちとして決着画面に移ります。
`No` または `ESC` で対局に戻ります。

### 決着画面

決着時は盤面中央に白いボックスが表示され、以下のように結果が表示されます。

- `Check Mate`
- `Stale Mate`
- `Draw`
- `Time Up`
- `Resign`

ボックス内には次のボタンがあります。

| ボタン | 内容 |
| --- | --- |
| `Replay` | 同じ条件で再対局 |
| `Top` | トップ画面へ戻る |
| `Save Record` | 保存名とメモを入力して棋譜保存 |

## 4. ルール

### 基本ルール

基本的な駒の動きはチェスと同じです。

| ルール | 内容 |
| --- | --- |
| 勝利条件 | 相手のキングをチェックメイトする |
| キャスリング | 有効 |
| アンパッサン | 有効 |
| ポーンの成り | 最終段で Queen / Rook / Bishop / Knight に成れる |
| ステイルメイト | 引き分け |
| 50 手ルール | 捕獲もポーン前進もない半手 100 回で引き分け |
| 三回同一局面 | 引き分け |

キングは捕獲されません。
キングを取る手は合法手として生成されず、勝敗はチェックメイトで決まります。

### 持ち駒ルール

相手の駒を取ると、その駒は自分の持ち駒になります。
自分の手番で、持ち駒を空きマスに打てます。

打ち駒の制約:

| 制約 | 内容 |
| --- | --- |
| 空きマスのみ | 既に駒があるマスには打てない |
| ポーン最終段打ち禁止 | 白ポーンは 8 段目、黒ポーンは 1 段目に打てない |
| 打ち駒即詰み禁止 | 持ち駒を打ったその手で相手を即チェックメイトにする手は不合法 |

持ち駒打ちでチェックを掛けること自体は合法です。
ただし、その打ち駒が即詰みになる場合は、ポーン以外の駒でも不合法になります。

### チェック表示

手を指した直後、相手のキングにチェックが掛かった場合は UI に `CHECK!` が表示されます。
保存棋譜にも各手ごとに `check: true/false` が記録されます。

## 5. 棋譜保存と再生

### 棋譜保存

対局中、または決着画面の `Save Record` を押すと、保存ダイアログが開きます。

入力できる項目:

- `Save name`
- `Memo`

下部には `Save` と `Cancel` があります。
保存中は `Saving...`、完了後は `Save Completed！` が表示され、ダイアログは自動で閉じます。

保存先:

```text
gamerecord\
```

保存ファイル名の例:

```text
gamerecord\20260529_153012_my_game.json
```

棋譜 JSON には主に以下が入ります。

- 対局モード
- 保存名
- メモ
- 開始時刻・保存時刻
- 先手色
- 持ち時間設定
- 結果
- 終了理由、通常 / 時間切れ / 投了
- 各手の移動元・移動先
- 成り情報
- 持ち駒打ち情報
- チェック判定
- 最終盤面

### 棋譜再生

トップ画面の `Replay GameRecord` から、`gamerecord\` 内の棋譜を一覧表示できます。

一覧画面でできること:

| 操作 | 内容 |
| --- | --- |
| 棋譜行をクリック | その棋譜を再生 |
| 左のチェックボックス | 棋譜を選択 |
| `Delete` | その 1 件を削除 |
| `Delete Selected` | 選択中の棋譜を削除 |
| `Delete All` | 棋譜を一括削除 |
| `Back` | トップ画面へ戻る |

削除前には確認ダイアログが表示されます。

再生画面のボタン:

| ボタン | 内容 |
| --- | --- |
| `<<` | 0 手目に戻る |
| `<` | 1 手戻す |
| `▶` | 1 秒に 1 手で自動再生 |
| `□` | 自動再生停止 |
| `>` | 1 手進める |
| `>>` | 最終局面へ進める |

## 6. AI 学習

### 学習の流れ

`python -m ai.train` は自己対戦学習を実行します。

流れ:

1. 現在のネットワークで自己対戦する
2. 各局面の `(状態, MCTS 後の方策, 勝敗)` を replay buffer に入れる
3. replay buffer からミニバッチをサンプルして学習する
4. 一定イテレーションごとにチェックポイントを保存する

学習進捗は 10 局ごとに次の形式で表示されます。

```text
10 / 200 steps
20 / 200 steps
```

ここでの `steps` は自己対戦局数です。

```text
total_iterations * num_self_play_games = 自己対戦の総局数
```

### モデル保存先

チェックポイントは次に保存されます。

```text
checkpoints\{run_name}\
```

例:

```text
checkpoints\run_quick_cpu\iter_00010.pt
checkpoints\run_quick_cpu\latest.pt
```

`latest.pt` は常に最新モデルです。
GUI で `--run run_quick_cpu` を指定した場合、この `latest.pt` を読みます。

### 自己対戦棋譜の保存

現在のコードでは、学習用の自己対戦棋譜も JSON で保存できます。
設定は `config/config.yaml` の `training` にあります。

```yaml
save_self_play_records: true
self_play_record_dir: "selfplay_records"
```

保存先:

```text
selfplay_records\{run_name}\
```

例:

```text
selfplay_records\run_quick_cpu\iter_00001_game_00001_step_000001.json
```

注意:

- 学習用サンプル自体はメモリ上の replay buffer に入ります。
- 自己対戦棋譜 JSON は分析・確認用の保存です。
- 既に起動済みの学習プロセスにはコード変更は反映されません。再起動後から有効です。

### 現在の軽量設定

現在の `config/config.yaml` は CPU で回しやすい軽量設定です。

主な値:

```yaml
ai:
  device: "cpu"
  mcts_simulations: 25
  gumbel_K: 8

network:
  num_res_blocks: 3
  channels: 64
  value_fc_size: 128

training:
  run_name: "run_quick_cpu"
  total_iterations: 20
  num_self_play_games: 10
  batch_size: 128
  num_epochs: 3
```

この設定では自己対戦総局数は次の通りです。

```text
20 * 10 = 200 局
```

### 10000 局学習の例

10000 局にしたい場合は、例えば次のようにします。

```yaml
training:
  run_name: "run_10000_games"
  total_iterations: 100
  num_self_play_games: 100
  save_every_n_iters: 10
  keep_last_n_checkpoints: 10
```

自己対戦総局数:

```text
100 * 100 = 10000 局
```

探索量を増やす場合:

```yaml
ai:
  mcts_simulations: 100
```

強くなりやすい一方、学習時間はかなり増えます。
対局時だけ読みを深くしたい場合は、学習後に `mcts_simulations` を上げて GUI を起動します。

## 7. 設定ファイル

設定ファイル:

```text
config\config.yaml
```

### `game`

| 項目 | 内容 |
| --- | --- |
| `max_moves` | 1 局の最大半手数 |
| `board_size` | 盤面サイズ。通常は `8` 固定 |

### `ai`

| 項目 | 内容 |
| --- | --- |
| `device` | `cpu` または `cuda` |
| `mcts_simulations` | 1 手あたりの MCTS シミュレーション数 |
| `gumbel_K` | Gumbel Sequential Halving の初期候補手数 |
| `c_puct` | PUCT の探索係数 |
| `dirichlet_alpha` | 自己対戦時のルートノイズ強度 |
| `dirichlet_eps` | ノイズ混合率 |
| `temperature` | 指し手サンプリング温度 |
| `temperature_threshold` | この手数以降は greedy に寄せる |
| `value_scale` | value head のスケール |

### `network`

| 項目 | 内容 |
| --- | --- |
| `num_res_blocks` | ResBlock 数 |
| `channels` | 中間チャンネル数 |
| `policy_channels` | policy head の中間チャンネル数 |
| `value_channels` | value head の中間チャンネル数 |
| `value_fc_size` | value head の全結合層サイズ |
| `input_planes` | 入力チャンネル数。現在は `28` |

### `training`

| 項目 | 内容 |
| --- | --- |
| `run_name` | 学習 run 名 |
| `total_iterations` | 学習イテレーション数 |
| `checkpoint_dir` | モデル保存先ルート |
| `save_every_n_iters` | 何イテレーションごとに保存するか |
| `keep_last_n_checkpoints` | 直近何個の `iter_XXXXX.pt` を残すか。`0` で全保持 |
| `resume` | `latest.pt` から再開するか |
| `batch_size` | 学習バッチサイズ |
| `lr` | 学習率 |
| `weight_decay` | L2 正則化 |
| `num_self_play_games` | 1 イテレーションあたりの自己対戦局数 |
| `num_epochs` | 1 イテレーションあたりの学習 epoch 数 |
| `buffer_size` | replay buffer の最大サンプル数 |
| `save_self_play_records` | 自己対戦棋譜 JSON を保存するか |
| `self_play_record_dir` | 自己対戦棋譜の保存先ルート |

### `ui`

| 項目 | 内容 |
| --- | --- |
| `window_width`, `window_height` | ウィンドウサイズ |
| `board_offset_x`, `board_offset_y` | 盤面画像の表示位置 |
| `board_display_width` | 盤面画像の表示幅 |
| `board_col_starts`, `board_row_starts` | 盤面画像内の 8x8 グリッド境界 |
| `board_frame_right`, `board_frame_bottom` | 盤面枠の右端・下端 |
| `board_sq_size` | 駒画像サイズ |
| `fps` | フレームレート |
| `animation_speed` | AI vs AI の表示待機時間 |
| `hand_panel_x` | 右サイドバーの X 座標 |
| `hand_piece_size` | 持ち駒アイコンサイズ |
| `font_size` | 基本フォントサイズ |
| `highlight_alpha` | ハイライト透明度 |
| `colors` | UI 色設定 |

## 8. ファイル構成

```text
Game02/
├── main.py
├── requirements.txt
├── chess_board.png
├── chess_pieces.png
├── config/
│   └── config.yaml
├── game/
│   ├── pieces.py
│   └── board.py
├── ai/
│   ├── network.py
│   ├── mcts.py
│   ├── agent.py
│   └── train.py
├── ui/
│   ├── assets.py
│   └── ui.py
├── tests/
│   └── test_hands_and_drops.py
├── checkpoints/
│   └── .gitkeep
├── gamerecord/
│   └── .gitkeep
└── selfplay_records/
    └── .gitkeep
```

主な役割:

| パス | 内容 |
| --- | --- |
| `game/pieces.py` | 駒種、色、Move、アクション ID 変換 |
| `game/board.py` | 盤面、合法手、持ち駒、詰み、引き分け判定 |
| `ai/network.py` | ResNet policy/value network |
| `ai/mcts.py` | Gumbel AlphaZero MCTS |
| `ai/agent.py` | GUI から使う AI エージェント |
| `ai/train.py` | 自己対戦学習 |
| `ui/assets.py` | 画像読み込み |
| `ui/ui.py` | Pygame UI、棋譜保存、再生、投了、時間管理 |

### Git 管理対象外のデータ

`.gitignore` で以下はアップロードされない設定です。

```text
checkpoints/*
gamerecord/*
selfplay_records/*
*.pt
*.pth
*.ckpt
*.onnx
```

各フォルダの `.gitkeep` だけを管理対象にしています。

## 9. テスト

持ち駒と打ち駒に関する回帰テストがあります。

```powershell
python -m unittest discover -s tests
```

現在確認している内容:

- 捕獲した駒が捕獲者の持ち駒に入る
- 白の持ち駒打ちは白い駒として置かれる
- 黒の持ち駒打ちは黒い駒として置かれる
- チェックになる持ち駒打ちは合法
- 即詰みになる持ち駒打ちは不合法
