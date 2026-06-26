"""
测试模块：使用模拟数据验证整个流程
=================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


def generate_mock_frame_data(fps=60, duration=2.0):
    """
    生成模拟的跳跃帧数据

    这模拟一个典型的跳跃过程：
    - 0.0-0.3s: 准备站立
    - 0.3-0.8s: 下蹲蓄力
    - 0.8-1.0s: 蹬伸起跳
    - 1.0-1.4s: 腾空
    - 1.4-2.0s: 落地

    返回：
        DataFrame，模拟的帧数据
    """
    n_frames = int(fps * duration)
    time = np.linspace(0, duration, n_frames)

    # 初始化数组
    knee_angle = np.zeros(n_frames)
    hip_angle = np.zeros(n_frames)
    center_y = np.zeros(n_frames)
    ankle_y = np.zeros(n_frames)
    arm_height = np.zeros(n_frames)
    trunk_angle = np.zeros(n_frames)

    for i, t in enumerate(time):
        if t < 0.3:
            # 准备站立
            knee_angle[i] = 175 + np.random.randn() * 2
            hip_angle[i] = 170 + np.random.randn() * 2
            center_y[i] = 0.5 + np.random.randn() * 0.005
            ankle_y[i] = 0.9
            arm_height[i] = 0.4
            trunk_angle[i] = 0

        elif t < 0.8:
            # 下蹲蓄力（膝角逐渐减小）
            progress = (t - 0.3) / 0.5
            knee_angle[i] = 175 - 55 * progress + np.random.randn() * 2
            hip_angle[i] = 170 - 40 * progress + np.random.randn() * 2
            center_y[i] = 0.5 + 0.15 * progress + np.random.randn() * 0.005  # 重心下降
            ankle_y[i] = 0.9
            arm_height[i] = 0.4 - 0.1 * progress  # 手臂下摆
            trunk_angle[i] = 15 * progress  # 躯干前倾

        elif t < 1.0:
            # 蹬伸起跳（膝角快速增大）
            progress = (t - 0.8) / 0.2
            knee_angle[i] = 120 + 55 * progress + np.random.randn() * 2
            hip_angle[i] = 130 + 45 * progress + np.random.randn() * 2
            center_y[i] = 0.65 - 0.25 * progress + np.random.randn() * 0.005  # 重心上升
            ankle_y[i] = 0.9 - 0.1 * progress  # 开始离地
            arm_height[i] = 0.3 + 0.5 * progress  # 手臂上摆
            trunk_angle[i] = 15 - 15 * progress  # 躯干回正

        elif t < 1.4:
            # 腾空
            progress = (t - 1.0) / 0.4
            # 抛物线
            h = 0.4 - 0.15 * progress  # 先上升后下降
            if progress > 0.5:
                h = 0.325 + 0.05 * (progress - 0.5)

            knee_angle[i] = 175 + np.random.randn() * 2
            hip_angle[i] = 175 + np.random.randn() * 2
            center_y[i] = h + np.random.randn() * 0.005
            ankle_y[i] = 0.8 - 0.3 * (1 - abs(progress - 0.5) * 2)  # 脚离地抛物线
            arm_height[i] = 0.8 - 0.1 * progress
            trunk_angle[i] = np.random.randn() * 2

        else:
            # 落地
            progress = (t - 1.4) / 0.6
            knee_angle[i] = 175 - 20 * (1 - progress) + np.random.randn() * 2
            hip_angle[i] = 175 - 15 * (1 - progress) + np.random.randn() * 2
            center_y[i] = 0.5 + 0.1 * (1 - progress) + np.random.randn() * 0.005
            ankle_y[i] = 0.9
            arm_height[i] = 0.5 - 0.1 * progress
            trunk_angle[i] = 5 * (1 - progress)

    # 构建 DataFrame
    df = pd.DataFrame({
        "frame": np.arange(n_frames),
        "time": time,
        "center_x": 0.5 + np.random.randn(n_frames) * 0.01,
        "center_y": center_y,
        "ankle_y": ankle_y,
        "head_y": center_y - 0.25,
        "wrist_y": 1 - arm_height,
        "arm_height": arm_height,
        "knee_angle": knee_angle,
        "left_knee_angle": knee_angle + np.random.randn(n_frames) * 3,
        "right_knee_angle": knee_angle - np.random.randn(n_frames) * 3,
        "hip_angle": hip_angle,
        "ankle_angle": 90 + np.random.randn(n_frames) * 5,
        "trunk_angle": trunk_angle,
        "knee_symmetry": np.abs(np.random.randn(n_frames) * 3),
        "visibility": 0.95 + np.random.randn(n_frames) * 0.02,
    })

    return df


def generate_mock_multi_subject(n_subjects=30, output_dir="mock_data"):
    """
    生成多人的模拟数据用于 XGBoost 训练

    参数：
        n_subjects: 人数
        output_dir: 输出目录

    返回：
        DataFrame，包含所有人的特征
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_features = []

    np.random.seed(42)

    for i in range(n_subjects):
        # 随机生成个体差异
        # 基础运动能力 (0-1)
        athletic_ability = np.random.beta(5, 3)  # 偏向中高水平

        # 8 项特征（有内在相关性）
        min_knee_angle = 100 + 40 * (1 - athletic_ability) + np.random.randn() * 10
        max_knee_angular_velocity = 600 + 400 * athletic_ability + np.random.randn() * 50
        hip_angle_change = 30 + 20 * athletic_ability + np.random.randn() * 5
        arm_swing_range = 0.2 + 0.4 * athletic_ability + np.random.randn() * 0.05
        arm_timing = -0.02 + np.random.randn() * 0.03  # 负=提前，正=滞后
        extension_time = 0.15 + 0.1 * (1 - athletic_ability) + np.random.randn() * 0.02
        trunk_angle_takeoff = 5 + 10 * (1 - athletic_ability) + np.random.randn() * 3
        takeoff_velocity = 0.3 + 0.4 * athletic_ability + np.random.randn() * 0.05

        # 跳跃高度（基于物理规律 + 各因素贡献）
        # 简化模型：h = f(v0, technique_factors)
        jump_height = (
            0.1  # 基础
            + 0.4 * takeoff_velocity  # 主要因素
            + 0.002 * max_knee_angular_velocity / 100  # 蹬伸速度
            + 0.3 * arm_swing_range  # 手臂摆动
            - 0.5 * abs(arm_timing)  # 时机偏差惩罚
            - 0.01 * abs(trunk_angle_takeoff)  # 前倾惩罚
            + np.random.randn() * 0.02  # 随机噪声
        )

        features = {
            "subject": f"subject_{i+1:03d}",
            "min_knee_angle": min_knee_angle,
            "max_knee_angular_velocity": max_knee_angular_velocity,
            "hip_angle_change": hip_angle_change,
            "arm_swing_range": arm_swing_range,
            "arm_timing": arm_timing,
            "extension_time": extension_time,
            "trunk_angle_takeoff": trunk_angle_takeoff,
            "takeoff_velocity": takeoff_velocity,
            "relative_jump_height": jump_height,
            "knee_symmetry": np.random.rand() * 10,
        }

        all_features.append(features)

        # 保存单独的 JSON
        json_path = output_dir / f"subject_{i+1:03d}_features.json"
        with open(json_path, "w") as f:
            json.dump(features, f, indent=2)

    # 汇总 CSV
    df = pd.DataFrame(all_features)
    df.to_csv(output_dir / "all_subjects.csv", index=False)

    print(f"已生成 {n_subjects} 人的模拟数据")
    print(f"保存位置: {output_dir}")
    print(f"\n数据摘要:")
    print(df.describe())

    return df


def test_full_pipeline():
    """测试完整流程"""
    print("=" * 60)
    print("       跳跃分析系统 - 模拟测试")
    print("=" * 60)

    # 1. 生成模拟帧数据
    print("\n[1/4] 生成模拟帧数据...")
    from core import FeatureExtractor
    df = generate_mock_frame_data(fps=60, duration=2.0)
    print(f"  生成 {len(df)} 帧数据")

    # 2. 提取特征
    print("\n[2/4] 提取跳跃特征...")
    extractor = FeatureExtractor(df, fps=60)
    summary, features = extractor.get_summary()
    print(summary)

    # 3. 生成可视化
    print("\n[3/4] 生成可视化图表...")
    from visualize import generate_all_plots
    output_dir = Path("test_output")
    generate_all_plots(df, features, output_dir)

    # 4. 测试多人建模（如果样本足够）
    print("\n[4/4] 测试 XGBoost + SHAP...")
    mock_data_dir = Path("mock_data")
    multi_df = generate_mock_multi_subject(30, mock_data_dir)

    try:
        import xgboost
        import shap
        from model import train_model, explain_with_shap, generate_report

        model, X, y, feature_names = train_model(multi_df)
        shap_values, importance = explain_with_shap(model, X, feature_names, output_dir)
        generate_report(multi_df, model, shap_values, importance, output_dir / "test_report.txt")
    except ImportError as e:
        print(f"  跳过 XGBoost/SHAP 测试（未安装）: {e}")

    print("\n" + "=" * 60)
    print("测试完成！输出文件:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()
