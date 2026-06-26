# 跳跃动作生物力学分析系统

基于 MediaPipe + XGBoost + SHAP 的弹跳摸高分析工具。

> **研究项目**: 基于计算机视觉与可解释机器学习的弹跳动作生物力学分析  
> **项目状态**: 开发中  
> **论文文档**: [docs/paper.md](docs/paper.md)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备测试视频

**拍摄规范（重要！）**：

| 参数 | 最佳值 | 说明 |
|------|--------|------|
| 角度 | 正侧面 90° | 避免肢体遮挡 |
| 高度 | 髋部高度 (~1m) | 减少透视畸变 |
| 距离 | 3-4 米 | 确保全身入镜 |
| 分辨率 | 1080p | 性价比最高 |
| 帧率 | 60fps | 捕捉快速动作 |
| 人物占比 | 画面高度 70-80% | 太小精度下降 |
| 背景 | 纯色/简洁 | 无其他人 |
| 服装 | 贴身运动服 | 与背景对比色 |

**示例视频下载**（免费）：
- Pexels: https://www.pexels.com/search/videos/vertical%20jump/
- Pixabay: https://pixabay.com/videos/search/jump/
- 搜索关键词: "vertical jump side view", "basketball jump slow motion"

### 3. 运行分析

```bash
python main.py your_video.mp4
```

可选参数：
```bash
python main.py video.mp4 --output results/    # 指定输出目录
python main.py video.mp4 --no-video           # 不生成标注视频
python main.py video.mp4 --no-plots           # 不生成图表
```

### 4. 查看结果

输出文件：
- `*_annotated.mp4` - 带骨骼标注的视频
- `*_raw_data.csv` - 每帧原始数据
- `*_features.json` - 提取的8项特征
- `*_report.txt` - 分析报告
- `joint_angles.png` - 关节角度变化图
- `velocities.png` - 速度/角速度图
- `trajectory.png` - 重心轨迹图
- `feature_radar.png` - 特征雷达图

## 文件结构

```
jump_analysis/
├── main.py           # 主程序入口
├── core.py           # 核心处理模块
├── visualize.py      # 可视化模块
├── config.py         # 配置参数
├── requirements.txt  # 依赖列表
└── README.md         # 说明文档
```

## 8项生物力学特征

1. **最小膝关节角度** - 下蹲深度
2. **膝关节蹬伸角速度** - 蹬地爆发力
3. **髋关节角度变化** - 伸髋幅度
4. **手臂摆动幅度** - 上肢助力
5. **手臂摆动时机** - 协调性
6. **蹬伸时间** - 发力节奏
7. **躯干前倾角** - 身体姿态
8. **离地瞬间速度** - 起跳速度

## 下一步：多人建模

收集多人数据后，使用 `model.py` 进行 XGBoost 建模和 SHAP 分析：

```bash
python model.py --data all_jumpers.csv
```

## 常见问题

**Q: 识别不到人？**
A: 检查光线、背景、人物是否完整入镜

**Q: 关节角度跳动？**
A: 调整 config.py 中的 smoothing_window

**Q: 阶段检测不准？**
A: 确保视频包含完整的站立→下蹲→起跳→落地过程

## 文档

| 文档 | 说明 |
|------|------|
| [docs/paper.md](docs/paper.md) | 完整论文（随项目迭代更新） |
| [docs/progress.md](docs/progress.md) | 项目进度跟踪 |
| [docs/data_collection_form.md](docs/data_collection_form.md) | 数据采集记录表 |
| [docs/consent_form.md](docs/consent_form.md) | 知情同意书模板 |

## 项目结构

```
jump_analysis/
├── main.py              # 主程序入口
├── core.py              # 核心处理模块（视频处理+特征提取）
├── model.py             # XGBoost建模 + SHAP分析
├── visualize.py         # 可视化模块
├── config.py            # 配置参数
├── test_mock.py         # 模拟测试
├── requirements.txt     # 依赖列表
├── README.md            # 说明文档
└── docs/                # 文档目录
    ├── paper.md         # 论文
    ├── progress.md      # 进度跟踪
    ├── data_collection_form.md  # 数据采集表
    └── consent_form.md  # 知情同意书
```

## 许可证

MIT License
