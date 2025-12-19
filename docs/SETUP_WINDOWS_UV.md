# SAM-Audio Windows + uv セットアップガイド

## 概要

SAM-Audio（Segment Anything Model for Audio）は、Meta（Facebook Research）が開発した音声分離のためのファウンデーションモデルです。テキスト、視覚、または時間範囲のプロンプトを使用して、複雑な音声ミックスから特定の音を分離できます。

このドキュメントは、Windows環境でuvパッケージマネージャーを使用してSAM-Audioをセットアップする手順を説明します。

---

## 前提条件

| 項目 | 要件 |
|------|------|
| OS | Windows 10/11 |
| GPU | NVIDIA GPU (CUDA 12.5〜12.6対応) |
| Python | 3.11以上 |
| パッケージマネージャー | uv |
| その他 | Git |

### uvのインストール（未インストールの場合）

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## セットアップ手順

### Step 1: リポジトリのクローン

```powershell
git clone https://github.com/facebookresearch/sam-audio.git
cd sam-audio
```

### Step 2: pyproject.toml の修正

uvでCUDA対応のPyTorchをインストールするため、`pyproject.toml`を修正します。

#### 2-1. Pythonバージョンの変更

`perception-models`がPython 3.11以上を要求するため、以下を変更：

```toml
# 変更前
requires-python = ">=3.10"

# 変更後
requires-python = ">=3.11"
```

#### 2-2. uv設定の追加

ファイル末尾に以下を追加：

```toml
# ========== uv設定（CUDA 12.6対応） ==========

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
# Git依存関係
dacvae = { git = "https://github.com/facebookresearch/dacvae.git" }
imagebind = { git = "https://github.com/facebookresearch/ImageBind.git" }
laion-clap = { git = "https://github.com/lematt1991/CLAP.git" }
perception-models = { git = "https://github.com/facebookresearch/perception_models.git", branch = "unpin-deps" }

# PyTorchパッケージ（CUDA 12.6インデックスから取得）
torch = { index = "pytorch-cu126" }
torchvision = { index = "pytorch-cu126" }
torchaudio = { index = "pytorch-cu126" }
# 注: torchcodecはWindows用CUDAビルドがないためPyPIから取得（指定不要）
```

#### 修正が必要な理由

| 修正項目 | 理由 |
|---------|------|
| `[[tool.uv.index]]` | PyPIのPyTorchはCPU版のみ。CUDA版はPyTorch公式インデックスから取得が必要 |
| `[tool.uv.sources]` | uvでGit依存関係を解決するために必要 |
| Python >=3.11 | perception-modelsの要件 |
| torchcodecを除外 | Windows用CUDA wheelが存在しないためPyPIにフォールバック |

### Step 3: 依存関係のインストール

```powershell
uv sync
```

初回実行時は依存関係のダウンロードに時間がかかります（約10〜15分）。

### Step 4: FFmpeg DLLの配置

torchcodecが動作するためにFFmpeg 6のDLL（共有ライブラリ）が必要です。

#### 方法A: OBS Studioからコピー（推奨）

OBS Studioがインストールされている場合、FFmpeg 6のDLLが含まれています：

```powershell
# FFmpegのインストールパスにコピー
Copy-Item "C:\Program Files\obs-studio\bin\64bit\avcodec-60.dll" "C:\Program Files (x86)\ffmpeg\"
Copy-Item "C:\Program Files\obs-studio\bin\64bit\avformat-60.dll" "C:\Program Files (x86)\ffmpeg\"
Copy-Item "C:\Program Files\obs-studio\bin\64bit\avutil-58.dll" "C:\Program Files (x86)\ffmpeg\"
Copy-Item "C:\Program Files\obs-studio\bin\64bit\swresample-4.dll" "C:\Program Files (x86)\ffmpeg\"
Copy-Item "C:\Program Files\obs-studio\bin\64bit\swscale-7.dll" "C:\Program Files (x86)\ffmpeg\"
```

#### 方法B: FFmpeg共有ビルドをダウンロード

https://www.gyan.dev/ffmpeg/builds/ から「shared」ビルドをダウンロードし、DLLをPATHの通ったディレクトリに配置。

### Step 5: HuggingFace認証

SAM-Audioのモデルはゲート付きリポジトリのため、アクセス申請が必要です。

1. HuggingFaceアカウントを作成（未作成の場合）
2. 以下のいずれかのモデルページでアクセス申請：
   - https://huggingface.co/facebook/sam-audio-small
   - https://huggingface.co/facebook/sam-audio-base
   - https://huggingface.co/facebook/sam-audio-large
3. 承認後、アクセストークンを取得してログイン：

```powershell
uv run huggingface-cli login
# プロンプトに従ってトークンを入力
```

---

## 実行方法

### サンプルスクリプト

リポジトリに`run_sample.py`が含まれています。テキストプロンプトと時間指定プロンプト（Span Prompting）の両方に対応しています。

### 実行コマンド

```powershell
# テキストプロンプトのみ
uv run python run_sample.py

# 時間指定プロンプト（1.0〜3.0秒の音を抽出）
uv run python run_sample.py --start 1.0 --end 3.0

# 除外モード（1.0〜3.0秒の音を除外）
uv run python run_sample.py --start 1.0 --end 3.0 --exclude

# カスタム入力・説明
uv run python run_sample.py --input your_audio.wav --description "A dog barking"
```

### コマンドラインオプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--start` | 開始時間（秒） | なし |
| `--end` | 終了時間（秒） | なし |
| `--exclude` | 指定範囲を除外 | False（抽出） |
| `--input` | 入力ファイル | `examples/assets/office.mp4` |
| `--description` | 分離対象の説明 | `A man speaking` |

### 時間指定プロンプト（Span Prompting）について

時間範囲を指定して、その区間に存在する音を分離する機能です。

```python
# 形式
anchors=[[[トークン, 開始秒, 終了秒]]]

# "+" は抽出、"-" は除外
anchors=[[["+", 1.0, 3.0]]]  # 1.0〜3.0秒の音を抽出
anchors=[[["-", 5.0, 7.0]]]  # 5.0〜7.0秒の音を除外

# 複数範囲の指定
anchors=[[["+", 0.5, 1.5], ["+", 3.0, 4.5]]]
```

### 出力ファイル

| ファイル | 内容 |
|---------|------|
| `target.wav` | 分離された音声（プロンプトに該当する音） |
| `residual.wav` | 残りの音声（プロンプトに該当しない音） |

---

## トラブルシューティング

### CUDA Out of Memory エラー

```
torch.OutOfMemoryError: CUDA out of memory.
```

**原因**: GPUメモリ不足（`sam-audio-large`は16GB以上必要）

**解決策**:
- `sam-audio-base`または`sam-audio-small`を使用
- メモリ最適化設定を追加：
  ```python
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
  torch.cuda.empty_cache()
  ```

### FFmpeg DLL not found

```
OSError: Could not find FFmpeg libraries
```

**原因**: torchcodecがFFmpegの共有ライブラリを見つけられない

**解決策**: Step 4を参照し、FFmpeg DLLをPATHの通ったディレクトリに配置

### HuggingFace 403 Forbidden

```
requests.exceptions.HTTPError: 403 Forbidden
```

**原因**: モデルへのアクセス権がない

**解決策**:
1. HuggingFaceでモデルへのアクセス申請
2. 承認後、`uv run huggingface-cli login`で認証

### perception-models インストールエラー

```
requires Python >=3.11
```

**原因**: Python 3.10以下を使用している

**解決策**: `pyproject.toml`の`requires-python`を`">=3.11"`に変更し、Python 3.11以上の環境を使用

---

## モデル一覧

### テキスト/時間プロンプト用モデル

| モデル | General SFX | Speech | Speaker | Music | Instr(wild) | Instr(pro) |
|--------|-------------|--------|---------|-------|-------------|------------|
| `sam-audio-small` | 3.62 | 3.99 | 3.12 | 4.11 | 3.56 | 4.24 |
| `sam-audio-base` | 3.28 | **4.25** | 3.57 | 3.87 | 3.66 | 4.27 |
| `sam-audio-large` | 3.50 | 4.03 | 3.60 | **4.22** | 3.66 | **4.49** |

※ スコアが高いほど精度が良い

### ビジュアルプロンプト用モデル（-tv版）

- `sam-audio-small-tv`
- `sam-audio-base-tv`
- `sam-audio-large-tv`

### 用途別おすすめモデル

| 用途 | おすすめモデル | 理由 |
|------|---------------|------|
| 話し声分離 | `sam-audio-base` | Speechスコア最高（4.25） |
| 音楽分離 | `sam-audio-large` | Musicスコア最高（4.22） |
| 楽器分離 | `sam-audio-large` | Instr(pro)スコア最高（4.49） |
| メモリ制限あり | `sam-audio-small` | 最軽量 |

---

## 参考リンク

- [SAM-Audio GitHub](https://github.com/facebookresearch/sam-audio)
- [SAM-Audio HuggingFace](https://huggingface.co/facebook/sam-audio-large)
- [uv Documentation](https://docs.astral.sh/uv/)
