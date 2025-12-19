"""SAM-Audio サンプル実行スクリプト

使用方法:
  # テキストプロンプトのみ
  uv run python run_sample.py

  # 時間指定プロンプト（Span Prompting）
  uv run python run_sample.py --start 1.0 --end 3.0

  # 除外モード（指定時間範囲を除外）
  uv run python run_sample.py --start 1.0 --end 3.0 --exclude
"""

import os
import argparse
import torch
import torchaudio
import gc
from sam_audio import SAMAudio, SAMAudioProcessor

# メモリ最適化設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser(description="SAM-Audio サンプル実行")
    parser.add_argument("--start", type=float, default=None, help="開始時間（秒）")
    parser.add_argument("--end", type=float, default=None, help="終了時間（秒）")
    parser.add_argument("--exclude", action="store_true", help="指定範囲を除外（デフォルトは抽出）")
    parser.add_argument("--input", type=str, default="examples/assets/office.mp4", help="入力ファイル")
    parser.add_argument("--description", type=str, default="A man speaking", help="分離対象の説明")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== SAM-Audio Sample ===")

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # GPUメモリをクリア
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # モデルとプロセッサをロード（baseモデルを使用してメモリ節約）
    model_name = "facebook/sam-audio-base"
    print(f"Loading model: {model_name}")
    model = SAMAudio.from_pretrained(model_name).to(device).eval()
    processor = SAMAudioProcessor.from_pretrained(model_name)
    print("Model loaded!")

    # 入力設定
    video_file = args.input
    description = args.description

    print(f"Input: {video_file}")
    print(f"Description: {description}")

    # 時間指定（Span Prompting）の設定
    anchors = None
    if args.start is not None and args.end is not None:
        token = "-" if args.exclude else "+"
        anchors = [[[token, args.start, args.end]]]
        mode = "除外" if args.exclude else "抽出"
        print(f"Span Prompting: {args.start}〜{args.end}秒を{mode}")

    # 処理
    print("Processing...")
    inputs = processor(
        audios=[video_file],
        descriptions=[description],
        anchors=anchors
    ).to(device)

    with torch.inference_mode():
        result = model.separate(inputs)

    # 保存
    sample_rate = processor.audio_sampling_rate
    torchaudio.save("target.wav", result.target[0].cpu(), sample_rate)
    torchaudio.save("residual.wav", result.residual[0].cpu(), sample_rate)

    print("=== Done! ===")
    print(f"Saved: target.wav (分離された音声)")
    print(f"Saved: residual.wav (残りの音声)")

if __name__ == "__main__":
    main()
