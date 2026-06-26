"""
可视化模块：生成分析图表
========================
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np
import pandas as pd
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_joint_angles(df, phases, output_path):
    """
    绘制关节角度随时间变化图

    参数：
        df: 帧数据 DataFrame
        phases: 阶段字典
        output_path: 输出文件路径
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    time = df["time"]

    # 1. 膝关节角度
    ax1 = axes[0]
    ax1.plot(time, df["knee_angle"], 'b-', linewidth=2, label='Knee Angle')
    ax1.axhline(y=170, color='gray', linestyle='--', alpha=0.5, label='Standing')
    ax1.set_ylabel('Knee Angle (degrees)')
    ax1.set_title('Joint Angles During Jump')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 标记阶段
    for name, frame in phases.items():
        if frame in df.index:
            t = df.loc[frame, "time"]
            ax1.axvline(x=t, color='red', linestyle=':', alpha=0.7)

    # 2. 髋关节角度
    ax2 = axes[1]
    ax2.plot(time, df["hip_angle"], 'g-', linewidth=2, label='Hip Angle')
    ax2.set_ylabel('Hip Angle (degrees)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    for name, frame in phases.items():
        if frame in df.index:
            t = df.loc[frame, "time"]
            ax2.axvline(x=t, color='red', linestyle=':', alpha=0.7)

    # 3. 重心高度
    ax3 = axes[2]
    ax3.plot(time, 1 - df["center_y"], 'r-', linewidth=2, label='Center Height')
    ax3.set_ylabel('Relative Height')
    ax3.set_xlabel('Time (seconds)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    for name, frame in phases.items():
        if frame in df.index:
            t = df.loc[frame, "time"]
            ax3.axvline(x=t, color='red', linestyle=':', alpha=0.7)
            ax3.annotate(name, xy=(t, ax3.get_ylim()[1]), fontsize=8, rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {output_path}")


def plot_velocities(df, phases, output_path):
    """
    绘制速度/角速度变化图
    """
    # 如果没有速度列，先计算
    if "knee_angular_velocity" not in df.columns:
        dt = df["time"].diff().mean()
        df = df.copy()
        df["knee_angular_velocity"] = df["knee_angle"].diff() / dt
        df["center_velocity"] = -df["center_y"].diff() / dt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    time = df["time"]

    # 1. 膝关节角速度
    ax1 = axes[0]
    ax1.plot(time, df["knee_angular_velocity"], 'b-', linewidth=2)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax1.set_ylabel('Knee Angular Velocity (deg/s)')
    ax1.set_title('Velocity Analysis')
    ax1.grid(True, alpha=0.3)

    # 标记最大角速度
    max_vel_idx = df["knee_angular_velocity"].idxmax()
    if max_vel_idx in df.index:
        max_t = df.loc[max_vel_idx, "time"]
        max_v = df.loc[max_vel_idx, "knee_angular_velocity"]
        ax1.annotate(f'Max: {max_v:.0f} deg/s',
                    xy=(max_t, max_v),
                    xytext=(max_t + 0.1, max_v),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='red'))

    # 2. 重心速度
    ax2 = axes[1]
    ax2.plot(time, df["center_velocity"], 'r-', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Center Velocity (relative)')
    ax2.set_xlabel('Time (seconds)')
    ax2.grid(True, alpha=0.3)

    # 标记离地时刻
    takeoff = phases.get("takeoff")
    if takeoff and takeoff in df.index:
        t = df.loc[takeoff, "time"]
        for ax in axes:
            ax.axvline(x=t, color='green', linestyle='--', linewidth=2, label='Takeoff')
        axes[0].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {output_path}")


def plot_trajectory(df, phases, output_path):
    """
    绘制重心轨迹图
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制轨迹
    x = df["center_x"]
    y = 1 - df["center_y"]  # 翻转y轴

    # 用颜色表示时间
    colors = np.linspace(0, 1, len(df))
    scatter = ax.scatter(x, y, c=colors, cmap='coolwarm', s=10, alpha=0.7)

    # 标记关键点
    markers = {
        "preparation": ("站立", "green", "o"),
        "squat_lowest": ("下蹲", "blue", "s"),
        "takeoff": ("离地", "orange", "^"),
        "peak": ("最高点", "red", "*"),
    }

    for phase_name, (label, color, marker) in markers.items():
        frame = phases.get(phase_name)
        if frame and frame in df.index:
            px = df.loc[frame, "center_x"]
            py = 1 - df.loc[frame, "center_y"]
            ax.scatter(px, py, c=color, s=200, marker=marker, label=label, zorder=5, edgecolors='black')

    ax.set_xlabel('Horizontal Position')
    ax.set_ylabel('Vertical Position (Height)')
    ax.set_title('Center of Mass Trajectory')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.colorbar(scatter, label='Time (normalized)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {output_path}")


def plot_feature_summary(features, output_path):
    """
    绘制特征雷达图
    """
    # 选择8个主要特征
    feature_names = [
        "Squat Depth\n(inv. knee angle)",
        "Extension Speed\n(knee ang. vel.)",
        "Hip Extension",
        "Arm Swing",
        "Arm Timing",
        "Extension Time",
        "Trunk Angle",
        "Takeoff Velocity"
    ]

    # 归一化特征值（示例范围）
    raw_values = [
        180 - features["min_knee_angle"],  # 反转，蹲得深=值大
        min(features["max_knee_angular_velocity"] / 10, 100),  # 缩放到0-100
        features["hip_angle_change"],
        features["arm_swing_range"] * 100,
        50 + features["arm_timing"] * 100,  # 居中
        100 - features["extension_time"] * 200,  # 反转
        90 - abs(features["trunk_angle_takeoff"]),  # 越接近0越好
        features["takeoff_velocity"] * 100,
    ]

    # 限制到 0-100
    values = [max(0, min(100, v)) for v in raw_values]
    values.append(values[0])  # 闭合

    # 雷达图
    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    ax.plot(angles, values, 'o-', linewidth=2, color='blue')
    ax.fill(angles, values, alpha=0.25, color='blue')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, size=10)
    ax.set_ylim(0, 100)
    ax.set_title('Jump Performance Profile', size=14, y=1.08)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {output_path}")


def generate_all_plots(df, features, output_dir):
    """
    生成所有分析图表

    参数：
        df: 帧数据
        features: 提取的特征
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phases = features["phases"]

    print("\n生成可视化图表...")
    plot_joint_angles(df, phases, output_dir / "joint_angles.png")
    plot_velocities(df, phases, output_dir / "velocities.png")
    plot_trajectory(df, phases, output_dir / "trajectory.png")
    plot_feature_summary(features, output_dir / "feature_radar.png")
    print("所有图表生成完成！")
