# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

SAM-Audio（Segment Anything Model for Audio）は、テキスト、視覚、または時間範囲のプロンプトを使用して音声から特定の音を分離するファウンデーションモデルです。Meta（Facebook Research）が開発。

## 開発コマンド

### インストール（uv使用）
```bash
uv sync
```

### リンター・フォーマッター
```bash
# フォーマットチェック
uv run ruff format --check .

# リント
uv run ruff check .

# 自動フォーマット
uv run ruff format .
```

### サンプル実行
```bash
# テキストプロンプトで実行
uv run python run_sample.py

# 時間指定プロンプト（Span Prompting）
uv run python run_sample.py --start 1.0 --end 3.0

# 除外モード
uv run python run_sample.py --start 1.0 --end 3.0 --exclude
```

### 評価実行
```bash
# デフォルト設定（instr-pro）で評価
uv run python eval/main.py

# 複数GPUで高速化
uv run torchrun --nproc_per_node=<ngpus> python eval/main.py

# 特定の設定で評価
uv run python eval/main.py --setting sfx speech music
```

## アーキテクチャ

### コアコンポーネント

**SAMAudio** (`sam_audio/model/model.py`)
- メインモデルクラス。ODEベースの拡散モデルで音源分離を実行
- `separate()`: 推論エントリーポイント。Batchを受け取り、target/residual音声を返す
- Flow Matchingを使用し、`torchdiffeq.odeint`で常微分方程式を解く

**SAMAudioProcessor** (`sam_audio/processor.py`)
- 入力の前処理を担当
- 音声、テキスト記述、アンカー（時間範囲）、マスクされた動画を`Batch`オブジェクトに変換
- `mask_videos()`: 視覚プロンプト用に動画とマスクを合成

**Batch** (`sam_audio/processor.py`)
- モデル入力をカプセル化するデータクラス
- `process_anchors()`: 時間範囲プロンプトをモデル入力形式に変換

### エンコーダー群 (`sam_audio/model/`)

- **DACVAE** (`codec.py`): 音声コーデック。波形⇔特徴量の変換
- **T5TextEncoder** (`text_encoder.py`): テキストプロンプトのエンコード
- **PerceptionEncoder** (`vision_encoder.py`): 視覚プロンプトのエンコード
- **DiT** (`transformer.py`): Diffusion Transformer。メイン生成モデル

### ランキングシステム (`sam_audio/ranking/`)

複数候補から最適な分離結果を選択：
- **ClapRanker**: テキストと音声の整合性スコア（CLAP）
- **ImageBindRanker**: 映像と音声の整合性スコア（視覚プロンプト用）
- **JudgeRanker**: 品質評価モデルによるスコア
- **EnsembleRanker**: 複数ランカーの重み付け結合

### 3種類のプロンプト方式

1. **テキストプロンプト**: 自然言語で分離したい音を記述
2. **視覚プロンプト**: 動画フレーム+マスクで視覚オブジェクトに関連する音を分離
3. **スパンプロンプト**: 時間範囲（`["+", start, end]`）で対象音の位置を指定

## 依存関係の特記事項

- `dacvae`, `imagebind`, `laion-clap`, `perception-models`: GitHubから直接インストール（`[tool.uv.sources]`で設定済み）
- Python 3.11以上必須（perception-modelsの要件）
- CUDA GPU推奨（CUDA 12.5〜12.6対応）
- PyTorchはCUDA版を`pytorch-cu126`インデックスから取得
- Hugging Faceでのモデルチェックポイントへのアクセス申請が必要
- Windows環境ではFFmpeg DLLが必要（torchcodec用）

## Ruff設定

- ターゲット: Python 3.11
- 有効ルール: B, C, E, W, F, I
- 無視: E501（行長）, E731（lambda代入）, C901（複雑度）, B006（可変デフォルト引数）

## uv設定

`pyproject.toml`にCUDA対応のuv設定を追加済み：
- `[[tool.uv.index]]`: PyTorch CUDA 12.6インデックス
- `[tool.uv.sources]`: Git依存関係とPyTorchソースの指定
