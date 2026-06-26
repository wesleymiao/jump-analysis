"""
XGBoost 建模 + SHAP 解释模块
============================

当收集了多人数据后，使用此模块进行因果分析
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_all_features(data_dir):
    """
    加载目录下所有人的特征数据

    参数：
        data_dir: 包含各人 *_features.json 的目录

    返回：
        DataFrame，每行是一个人的特征
    """
    data_dir = Path(data_dir)
    all_data = []

    for json_file in data_dir.glob("*_features.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            features = json.load(f)

        # 移除 phases 字段
        if "phases" in features:
            del features["phases"]

        features["subject"] = json_file.stem.replace("_features", "")
        all_data.append(features)

    if not all_data:
        raise ValueError(f"目录 {data_dir} 中没有找到特征文件")

    return pd.DataFrame(all_data)


def train_model(df, target_col="relative_jump_height"):
    """
    训练 XGBoost 模型

    参数：
        df: 特征数据
        target_col: 目标变量列名

    返回：
        model, X, y, feature_names
    """
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score

    # 特征列
    feature_cols = [
        "min_knee_angle",
        "max_knee_angular_velocity",
        "hip_angle_change",
        "arm_swing_range",
        "arm_timing",
        "extension_time",
        "trunk_angle_takeoff",
        "takeoff_velocity",
    ]

    # 检查列是否存在
    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) < len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        print(f"警告：缺少特征列: {missing}")

    X = df[available_cols]
    y = df[target_col]

    # 训练模型
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )

    # 交叉验证
    scores = cross_val_score(model, X, y, cv=min(5, len(df)), scoring="r2")
    print(f"交叉验证 R² 分数: {scores.mean():.3f} (+/- {scores.std()*2:.3f})")

    # 在全部数据上训练
    model.fit(X, y)

    return model, X, y, available_cols


def explain_with_shap(model, X, feature_names, output_dir):
    """
    使用 SHAP 解释模型

    参数：
        model: 训练好的模型
        X: 特征矩阵
        feature_names: 特征名称列表
        output_dir: 输出目录
    """
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 特征名称映射（英文→中文）
    name_map = {
        "min_knee_angle": "膝关节最小角度",
        "max_knee_angular_velocity": "膝关节蹬伸角速度",
        "hip_angle_change": "髋关节角度变化",
        "arm_swing_range": "手臂摆动幅度",
        "arm_timing": "手臂摆动时机",
        "extension_time": "蹬伸时间",
        "trunk_angle_takeoff": "躯干前倾角",
        "takeoff_velocity": "离地瞬间速度",
    }

    # 1. 特征重要性汇总图
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP 汇总图已保存: {output_dir / 'shap_summary.png'}")

    # 2. 特征重要性条形图
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP 重要性图已保存: {output_dir / 'shap_importance.png'}")

    # 3. 计算平均绝对 SHAP 值（特征排名）
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "feature_cn": [name_map.get(f, f) for f in feature_names],
        "mean_abs_shap": mean_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n特征重要性排名：")
    print("=" * 50)
    for i, row in feature_importance.iterrows():
        print(f"  {row['feature_cn']}: {row['mean_abs_shap']:.4f}")

    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    return shap_values, feature_importance


def explain_individual(model, X, idx, feature_names, output_dir):
    """
    解释单个人的预测结果

    参数：
        model: 模型
        X: 特征矩阵
        idx: 样本索引
        feature_names: 特征名称
        output_dir: 输出目录
    """
    import shap

    output_dir = Path(output_dir)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 瀑布图
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X.iloc[idx],
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_waterfall_{idx}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"个人解释图已保存: {output_dir / f'shap_waterfall_{idx}.png'}")


def generate_report(df, model, shap_values, feature_importance, output_path):
    """
    生成完整的分析报告
    """
    report = """
================================================================================
                        跳跃动作因果分析报告
================================================================================

【数据概况】
  样本数量: {n_samples} 人
  特征数量: {n_features} 项

【模型性能】
  模型类型: XGBoost 回归
  训练完成: 是

【特征重要性排名】
  (数值越大，对跳跃高度的影响越大)

{importance_table}

【主要发现】
  1. 最重要的因素: {top1_cn}
     - 平均影响幅度: {top1_value:.4f}

  2. 第二重要因素: {top2_cn}
     - 平均影响幅度: {top2_value:.4f}

  3. 第三重要因素: {top3_cn}
     - 平均影响幅度: {top3_value:.4f}

【训练建议】
  基于以上分析，提高弹跳高度应优先训练：
  - {top1_cn}：是影响弹跳的最关键因素
  - {top2_cn}：是第二关键因素
  - {top3_cn}：是第三关键因素

================================================================================
"""
    # 格式化重要性表格
    importance_table = ""
    for i, row in feature_importance.iterrows():
        importance_table += f"  {row['feature_cn']:20s}  {row['mean_abs_shap']:.4f}\n"

    top3 = feature_importance.head(3)

    report = report.format(
        n_samples=len(df),
        n_features=len(feature_importance),
        importance_table=importance_table,
        top1_cn=top3.iloc[0]["feature_cn"],
        top1_value=top3.iloc[0]["mean_abs_shap"],
        top2_cn=top3.iloc[1]["feature_cn"],
        top2_value=top3.iloc[1]["mean_abs_shap"],
        top3_cn=top3.iloc[2]["feature_cn"],
        top3_value=top3.iloc[2]["mean_abs_shap"],
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n分析报告已保存: {output_path}")
    print(report)


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="XGBoost 建模 + SHAP 分析")
    parser.add_argument("--data", required=True, help="特征数据目录或CSV文件")
    parser.add_argument("--output", "-o", default="model_output", help="输出目录")
    parser.add_argument("--target", default="relative_jump_height", help="目标变量")

    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("加载数据...")
    if data_path.is_dir():
        df = load_all_features(data_path)
    else:
        df = pd.read_csv(data_path)

    print(f"  加载了 {len(df)} 个样本")

    # 训练模型
    print("\n训练模型...")
    model, X, y, feature_names = train_model(df, args.target)

    # SHAP 解释
    print("\nSHAP 分析...")
    shap_values, feature_importance = explain_with_shap(model, X, feature_names, output_dir)

    # 生成报告
    generate_report(df, model, shap_values, feature_importance, output_dir / "analysis_report.txt")

    # 为每个人生成个人解释
    print("\n生成个人解释图...")
    for idx in range(min(5, len(df))):  # 最多生成5个
        explain_individual(model, X, idx, feature_names, output_dir)

    print("\n分析完成！")


if __name__ == "__main__":
    main()
