#!/usr/bin/env python3
"""
跳跃分析主程序
==============

使用方法：
    python main.py <视频路径> [--output 输出目录]

示例：
    python main.py sample_jump.mp4
    python main.py sample_jump.mp4 --output results/
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="跳跃动作生物力学分析")
    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument("--output", "-o", default="output", help="输出目录 (默认: output)")
    parser.add_argument("--no-video", action="store_true", help="不生成标注视频")
    parser.add_argument("--no-plots", action="store_true", help="不生成图表")

    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output)

    if not video_path.exists():
        print(f"错误：视频文件不存在: {video_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("     跳跃动作生物力学分析系统")
    print("=" * 50)
    print(f"\n输入视频: {video_path}")
    print(f"输出目录: {output_dir}\n")

    # 导入模块
    from core import JumpAnalyzer, FeatureExtractor

    # 1. 处理视频
    print("[步骤 1/4] 视频处理 + 姿态估计")
    analyzer = JumpAnalyzer()

    output_video = None if args.no_video else output_dir / f"{video_path.stem}_annotated.mp4"
    df = analyzer.process_video(video_path, output_video)

    if df.empty:
        print("错误：未能识别到任何姿态，请检查视频质量")
        sys.exit(1)

    # 保存原始数据
    csv_path = output_dir / f"{video_path.stem}_raw_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"原始数据已保存: {csv_path}")

    # 2. 特征提取
    print("\n[步骤 2/4] 特征提取")
    extractor = FeatureExtractor(df, fps=analyzer.fps)
    summary, features = extractor.get_summary()
    print(summary)

    # 保存特征
    import json
    features_for_json = {k: v for k, v in features.items() if k != "phases"}
    features_for_json["phases"] = {k: int(v) for k, v in features["phases"].items()}
    features_path = output_dir / f"{video_path.stem}_features.json"
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(features_for_json, f, indent=2, ensure_ascii=False)
    print(f"特征数据已保存: {features_path}")

    # 3. 生成图表
    if not args.no_plots:
        print("\n[步骤 3/4] 生成可视化图表")
        from visualize import generate_all_plots
        generate_all_plots(df, features, output_dir)

    # 4. 生成报告
    print("\n[步骤 4/4] 生成分析报告")
    report_path = output_dir / f"{video_path.stem}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"报告已保存: {report_path}")

    print("\n" + "=" * 50)
    print("分析完成！输出文件：")
    print("=" * 50)
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
