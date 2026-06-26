"""
核心处理模块：视频处理 + 姿态估计 + 特征提取
============================================
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import math
from config import MEDIAPIPE_CONFIG, LANDMARKS, FEATURE_CONFIG


class JumpAnalyzer:
    """跳跃动作分析器"""

    def __init__(self):
        """初始化 MediaPipe Pose"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=MEDIAPIPE_CONFIG["model_complexity"],
            enable_segmentation=MEDIAPIPE_CONFIG["enable_segmentation"],
            min_detection_confidence=MEDIAPIPE_CONFIG["min_detection_confidence"],
            min_tracking_confidence=MEDIAPIPE_CONFIG["min_tracking_confidence"],
            smooth_landmarks=MEDIAPIPE_CONFIG["smooth_landmarks"],
        )

        self.frame_data = []
        self.fps = 30

    def calculate_angle(self, a, b, c):
        """
        计算三个点形成的角度（b是顶点）

        参数：
            a, b, c: MediaPipe landmark 对象

        返回：
            角度（度）
        """
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])

        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))
        return angle

    def calculate_distance(self, a, b):
        """计算两点之间的距离"""
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def extract_frame_features(self, landmarks, frame_num):
        """
        从单帧关键点提取特征

        参数：
            landmarks: MediaPipe pose landmarks
            frame_num: 帧编号

        返回：
            特征字典
        """
        lm = landmarks.landmark

        # 获取关键点
        left_shoulder = lm[LANDMARKS["LEFT_SHOULDER"]]
        right_shoulder = lm[LANDMARKS["RIGHT_SHOULDER"]]
        left_hip = lm[LANDMARKS["LEFT_HIP"]]
        right_hip = lm[LANDMARKS["RIGHT_HIP"]]
        left_knee = lm[LANDMARKS["LEFT_KNEE"]]
        right_knee = lm[LANDMARKS["RIGHT_KNEE"]]
        left_ankle = lm[LANDMARKS["LEFT_ANKLE"]]
        right_ankle = lm[LANDMARKS["RIGHT_ANKLE"]]
        left_wrist = lm[LANDMARKS["LEFT_WRIST"]]
        right_wrist = lm[LANDMARKS["RIGHT_WRIST"]]
        nose = lm[LANDMARKS["NOSE"]]

        # 1. 重心位置（髋关节中点）
        center_x = (left_hip.x + right_hip.x) / 2
        center_y = (left_hip.y + right_hip.y) / 2

        # 2. 膝关节角度
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

        # 3. 髋关节角度
        left_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self.calculate_angle(right_shoulder, right_hip, right_knee)
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2

        # 4. 踝关节角度
        left_foot = lm[LANDMARKS["LEFT_FOOT_INDEX"]]
        right_foot = lm[LANDMARKS["RIGHT_FOOT_INDEX"]]
        left_ankle_angle = self.calculate_angle(left_knee, left_ankle, left_foot)
        right_ankle_angle = self.calculate_angle(right_knee, right_ankle, right_foot)
        avg_ankle_angle = (left_ankle_angle + right_ankle_angle) / 2

        # 5. 手腕高度（用于判断摸高）
        left_wrist_y = left_wrist.y
        right_wrist_y = right_wrist.y
        min_wrist_y = min(left_wrist_y, right_wrist_y)  # y越小越高

        # 6. 脚踝高度（用于判断离地）
        ankle_y = (left_ankle.y + right_ankle.y) / 2

        # 7. 躯干前倾角（肩-髋连线与垂直线的夹角）
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        trunk_angle = math.degrees(math.atan2(
            shoulder_center_x - center_x,
            center_y - shoulder_center_y  # y轴向下为正
        ))

        # 8. 手臂位置
        arm_height = 1 - min_wrist_y  # 转换为越高越大

        # 9. 身体对称性（左右膝关节角度差）
        knee_symmetry = abs(left_knee_angle - right_knee_angle)

        # 10. 头部高度
        head_y = nose.y

        # 11. 置信度（取关键点的平均可见度）
        key_points = [left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]
        avg_visibility = sum(p.visibility for p in key_points) / len(key_points)

        return {
            "frame": frame_num,
            "time": frame_num / self.fps,
            # 位置
            "center_x": center_x,
            "center_y": center_y,
            "ankle_y": ankle_y,
            "head_y": head_y,
            "wrist_y": min_wrist_y,
            "arm_height": arm_height,
            # 角度
            "knee_angle": avg_knee_angle,
            "left_knee_angle": left_knee_angle,
            "right_knee_angle": right_knee_angle,
            "hip_angle": avg_hip_angle,
            "ankle_angle": avg_ankle_angle,
            "trunk_angle": trunk_angle,
            # 对称性
            "knee_symmetry": knee_symmetry,
            # 置信度
            "visibility": avg_visibility,
        }

    def process_video(self, video_path, output_video_path=None):
        """
        处理视频，提取每帧特征

        参数：
            video_path: 输入视频路径
            output_video_path: 输出标注视频路径（可选）

        返回：
            DataFrame，包含所有帧的特征
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频信息: {width}x{height}, {self.fps:.1f}fps, {total_frames}帧")

        # 准备输出视频
        out = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, self.fps, (width, height))

        self.frame_data = []
        frame_num = 0

        print("处理中...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # BGR 转 RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                # 提取特征
                features = self.extract_frame_features(results.pose_landmarks, frame_num)
                self.frame_data.append(features)

                # 绘制骨骼
                if out:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                    # 显示实时数据
                    cv2.putText(frame, f"Knee: {features['knee_angle']:.1f}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Hip: {features['hip_angle']:.1f}",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Frame: {frame_num}",
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if out:
                out.write(frame)

            frame_num += 1

            # 显示进度
            if frame_num % 30 == 0:
                print(f"  已处理 {frame_num}/{total_frames} 帧 ({100*frame_num/total_frames:.1f}%)")

        cap.release()
        if out:
            out.release()

        print(f"完成！共识别 {len(self.frame_data)} 帧")

        return pd.DataFrame(self.frame_data)


class FeatureExtractor:
    """从时序数据提取跳跃特征"""

    def __init__(self, df, fps=30):
        """
        参数：
            df: process_video 返回的 DataFrame
            fps: 视频帧率
        """
        self.df = df.copy()
        self.fps = fps
        self._compute_velocities()

    def _smooth(self, series, window=5):
        """滑动平均平滑"""
        return series.rolling(window=window, center=True, min_periods=1).mean()

    def _compute_velocities(self):
        """计算各项速度/角速度"""
        dt = 1.0 / self.fps

        # 平滑原始数据
        window = FEATURE_CONFIG["smoothing_window"]
        self.df["knee_angle_smooth"] = self._smooth(self.df["knee_angle"], window)
        self.df["center_y_smooth"] = self._smooth(self.df["center_y"], window)
        self.df["ankle_y_smooth"] = self._smooth(self.df["ankle_y"], window)
        self.df["arm_height_smooth"] = self._smooth(self.df["arm_height"], window)

        # 计算速度
        self.df["knee_angular_velocity"] = self.df["knee_angle_smooth"].diff() / dt
        self.df["center_velocity"] = -self.df["center_y_smooth"].diff() / dt  # 负号：y向下为正，转为向上为正
        self.df["ankle_velocity"] = -self.df["ankle_y_smooth"].diff() / dt
        self.df["arm_velocity"] = self.df["arm_height_smooth"].diff() / dt

    def detect_phases(self):
        """
        检测跳跃阶段

        返回：
            dict，包含各阶段的帧索引
        """
        # 1. 找到下蹲最低点（膝关节角度最小）
        squat_frame = self.df["knee_angle_smooth"].idxmin()

        # 2. 找到离地时刻（脚踝速度突然增大）
        # 从下蹲最低点开始往后找
        post_squat = self.df.loc[squat_frame:]
        ankle_vel = post_squat["ankle_velocity"]

        # 找到脚踝向上速度超过阈值的第一帧
        takeoff_candidates = ankle_vel[ankle_vel > 0.5].index
        takeoff_frame = takeoff_candidates[0] if len(takeoff_candidates) > 0 else squat_frame + 5

        # 3. 找到最高点（重心y最小）
        post_takeoff = self.df.loc[takeoff_frame:]
        peak_frame = post_takeoff["center_y_smooth"].idxmin()

        # 4. 找到准备阶段开始（下蹲开始前的站立状态）
        pre_squat = self.df.loc[:squat_frame]
        standing_candidates = pre_squat[pre_squat["knee_angle_smooth"] > 165].index
        prep_frame = standing_candidates[-1] if len(standing_candidates) > 0 else 0

        return {
            "preparation": prep_frame,
            "squat_lowest": squat_frame,
            "takeoff": takeoff_frame,
            "peak": peak_frame,
        }

    def extract_jump_features(self):
        """
        提取跳跃的关键生物力学特征

        返回：
            dict，包含8个主要特征
        """
        phases = self.detect_phases()

        squat_frame = phases["squat_lowest"]
        takeoff_frame = phases["takeoff"]
        peak_frame = phases["peak"]
        prep_frame = phases["preparation"]

        # 1. 最小膝关节角度（下蹲深度）
        min_knee_angle = self.df.loc[squat_frame, "knee_angle"]

        # 2. 膝关节蹬伸角速度（最大值）
        extension_phase = self.df.loc[squat_frame:takeoff_frame]
        max_knee_angular_velocity = extension_phase["knee_angular_velocity"].max()

        # 3. 髋关节角度变化
        hip_angle_squat = self.df.loc[squat_frame, "hip_angle"]
        hip_angle_takeoff = self.df.loc[takeoff_frame, "hip_angle"]
        hip_angle_change = hip_angle_takeoff - hip_angle_squat

        # 4. 手臂摆动幅度
        arm_at_squat = self.df.loc[squat_frame, "arm_height"]
        arm_at_peak = self.df.loc[peak_frame, "arm_height"]
        arm_swing_range = arm_at_peak - arm_at_squat

        # 5. 手臂摆动时机（手臂最高速度相对于离地的时间差）
        arm_max_vel_frame = self.df.loc[squat_frame:peak_frame, "arm_velocity"].idxmax()
        arm_timing = (arm_max_vel_frame - takeoff_frame) / self.fps

        # 6. 蓄力时间（从最低点到离地）
        extension_time = (takeoff_frame - squat_frame) / self.fps

        # 7. 躯干前倾角（离地时）
        trunk_angle_takeoff = self.df.loc[takeoff_frame, "trunk_angle"]

        # 8. 离地瞬间重心速度
        takeoff_velocity = self.df.loc[takeoff_frame, "center_velocity"]

        # 额外：跳跃高度（相对值）
        center_at_standing = self.df.loc[prep_frame, "center_y"]
        center_at_peak = self.df.loc[peak_frame, "center_y"]
        relative_jump_height = center_at_standing - center_at_peak  # y向下为正，所以是减法

        # 膝关节对称性
        knee_symmetry = self.df.loc[squat_frame, "knee_symmetry"]

        return {
            # 8 个主要特征
            "min_knee_angle": min_knee_angle,
            "max_knee_angular_velocity": max_knee_angular_velocity,
            "hip_angle_change": hip_angle_change,
            "arm_swing_range": arm_swing_range,
            "arm_timing": arm_timing,
            "extension_time": extension_time,
            "trunk_angle_takeoff": trunk_angle_takeoff,
            "takeoff_velocity": takeoff_velocity,
            # 附加指标
            "relative_jump_height": relative_jump_height,
            "knee_symmetry": knee_symmetry,
            # 阶段信息
            "phases": phases,
        }

    def get_summary(self):
        """生成分析摘要"""
        features = self.extract_jump_features()
        phases = features["phases"]

        summary = f"""
========================================
        跳跃动作分析报告
========================================

【阶段检测】
  准备站立：第 {phases['preparation']} 帧
  下蹲最低：第 {phases['squat_lowest']} 帧
  离地时刻：第 {phases['takeoff']} 帧
  最高点：  第 {phases['peak']} 帧

【8项生物力学特征】
  1. 最小膝关节角度：  {features['min_knee_angle']:.1f}° (越小=蹲得越深)
  2. 膝关节蹬伸角速度：{features['max_knee_angular_velocity']:.1f}°/s (越大=蹬地越猛)
  3. 髋关节角度变化：  {features['hip_angle_change']:.1f}° (伸髋幅度)
  4. 手臂摆动幅度：    {features['arm_swing_range']:.3f} (相对值)
  5. 手臂摆动时机：    {features['arm_timing']*1000:.1f}ms (相对离地，负=提前，正=滞后)
  6. 蹬伸时间：        {features['extension_time']*1000:.1f}ms
  7. 躯干前倾角：      {features['trunk_angle_takeoff']:.1f}° (正=前倾)
  8. 离地瞬间速度：    {features['takeoff_velocity']:.3f} (相对值)

【附加指标】
  相对跳跃高度：{features['relative_jump_height']:.3f} (相对值)
  膝关节对称性：{features['knee_symmetry']:.1f}° (越小越对称)

========================================
"""
        return summary, features
