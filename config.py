"""
跳跃分析配置文件
================
调整这些参数以优化识别效果
"""

# ===== 拍摄规范 =====
RECOMMENDED_SETTINGS = {
    "角度": "正侧面 90度 (可接受 75-105度)",
    "相机高度": "1米 (髋部高度)",
    "拍摄距离": "3-4米",
    "分辨率": "1080p (1920x1080)",
    "帧率": "60fps (最低30fps)",
    "人物占比": "画面高度的 70-80%",
    "背景": "纯色或简洁，无其他人",
    "光线": "明亮均匀，避免逆光",
    "服装": "贴身运动服，与背景对比色",
}

# ===== MediaPipe 配置 =====
MEDIAPIPE_CONFIG = {
    "model_complexity": 2,          # 0=轻量, 1=中等, 2=精确(推荐)
    "min_detection_confidence": 0.5,  # 检测置信度阈值
    "min_tracking_confidence": 0.5,   # 追踪置信度阈值
    "enable_segmentation": False,     # 是否启用人体分割
    "smooth_landmarks": True,         # 平滑关键点
}

# ===== 关键点索引 (MediaPipe Pose) =====
LANDMARKS = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29,
    "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}

# ===== 特征提取配置 =====
FEATURE_CONFIG = {
    # 角度阈值（度）
    "knee_angle_standing": 170,      # 站立时膝关节角度
    "knee_angle_squat_min": 90,      # 下蹲最小角度

    # 速度阈值
    "min_angular_velocity": 50,      # 最小有效角速度 (度/秒)

    # 时间阈值（秒）
    "min_squat_duration": 0.1,       # 最小蓄力时间
    "max_squat_duration": 1.0,       # 最大蓄力时间

    # 平滑参数
    "smoothing_window": 5,           # 滑动平均窗口大小
}

# ===== 输出配置 =====
OUTPUT_CONFIG = {
    "save_video": True,              # 是否保存标注视频
    "save_csv": True,                # 是否保存CSV数据
    "save_plots": True,              # 是否保存图表
    "video_codec": "mp4v",           # 视频编码
    "plot_dpi": 150,                 # 图表分辨率
}
