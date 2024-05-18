# video_processor.py
import cv2
import mediapipe as mp
import os
import csv
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 定义全局斜率阈值
threshold = 2
hip_threshold = 0.1
shoulder_threshold = 0.1

def calculate_slope(landmarks, point1, point2):
    x1 = landmarks[point1].x
    y1 = landmarks[point1].y
    x2 = landmarks[point2].x
    y2 = landmarks[point2].y
    slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
    return slope

def adjust_color(frame):
    frame[:, :, 0] = np.clip(frame[:, :, 0] * 0.5, 0, 255)
    return frame

def generate_csv_file(clip_name, knee_slope_triggered, output_folder):
    if knee_slope_triggered:
        csv_filename = os.path.join(output_folder, f"{clip_name}_start_frame.csv")
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(['部位', '膝蓋'])
            writer.writerow([])
            writer.writerow(['錯誤原因', '', '膝蓋內夾 關節壓力大'])
            writer.writerow(['改善建議', '', '保持外旋張力 臀部持續發力'])
            writer.writerow(['正確動作', '', '膝關節盡量不要內縮到後腳跟裡面'])

def generate_csv_file2(clip_name, hip_slope_triggered, output_folder):
    if hip_slope_triggered:
        csv_filename = os.path.join(output_folder, f"{clip_name}_start_frame.csv")
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(['部位', '臀部'])
            writer.writerow([])
            writer.writerow(['錯誤原因', '', '重心歪斜 左右肌力不平衡'])
            writer.writerow(['改善建議', '', '在訓練中加入單邊訓練'])
            writer.writerow(['正確動作', '', '左右腳須以相同速率上升下降'])

def generate_csv_file3(clip_name, shoulder_slope_triggered, output_folder):
    if shoulder_slope_triggered:
        csv_filename = os.path.join(output_folder, f"{clip_name}_start_frame.csv")
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(['部位', '上肢'])
            writer.writerow([])
            writer.writerow(['錯誤原因', '', '上肢對於槓鈴鎖定不完全'])
            writer.writerow(['改善建議', '', '增加背部訓練 增強鎖定力度'])
            writer.writerow(['正確動作', '', '肩關節維持水平 槓子才可以保持穩定'])

def find_clip_ranges(input_path, landmarks_to_check, clip_name, output_folder):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clip_ranges = []
    current_clip_range = None
    current_frame = 0
    knee_slope_triggered = False
    csv_generated = False

    while cap.isOpened() and current_frame < total_frames:
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                slope = abs(calculate_slope(landmarks, landmarks_to_check[0], landmarks_to_check[1]))
                if current_clip_range is None and slope < threshold:
                    current_clip_range = {'start_frame': current_frame, 'slope': slope}
                    knee_slope_triggered = True
                elif current_clip_range is not None and slope > threshold:
                    current_clip_range['end_frame'] = current_frame
                    clip_ranges.append(current_clip_range)
                    current_clip_range = None
                    knee_slope_triggered = False
                    csv_generated = False
                if knee_slope_triggered and not csv_generated:
                    generate_csv_file(clip_name, knee_slope_triggered, output_folder)
                    csv_generated = True
            current_frame += 1
        else:
            break

    if current_clip_range is not None:
        current_clip_range['end_frame'] = current_frame
        clip_ranges.append(current_clip_range)

    cap.release()
    cv2.destroyAllWindows()
    return clip_ranges

def find_clip_ranges2(input_path, hip_clip_landmarks, clip_name, output_folder):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    hip_clip_ranges = []
    current_clip_range = None
    current_frame = 0
    hip_slope_triggered = False
    csv_generated = False

    while cap.isOpened() and current_frame < total_frames:
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                slope = abs(calculate_slope(landmarks, hip_clip_landmarks[0], hip_clip_landmarks[1]))
                if current_clip_range is None and slope > hip_threshold:
                    current_clip_range = {'start_frame': current_frame, 'slope': slope}
                    hip_slope_triggered = True
                elif current_clip_range is not None and slope < hip_threshold:
                    current_clip_range['end_frame'] = current_frame
                    hip_clip_ranges.append(current_clip_range)
                    current_clip_range = None
                    hip_slope_triggered = False
                    csv_generated = False
                if hip_slope_triggered and not csv_generated:
                    generate_csv_file2(clip_name, hip_slope_triggered, output_folder)
                    csv_generated = True
            current_frame += 1
        else:
            break

    if current_clip_range is not None:
        current_clip_range['end_frame'] = current_frame
        hip_clip_ranges.append(current_clip_range)

    cap.release()
    cv2.destroyAllWindows()
    return hip_clip_ranges

def find_clip_ranges3(input_path, shoulder_clip_landmarks, clip_name, output_folder):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    shoulder_clip_ranges = []
    current_clip_range = None
    current_frame = 0
    shoulder_slope_triggered = False
    csv_generated = False

    while cap.isOpened() and current_frame < total_frames:
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                slope = abs(calculate_slope(landmarks, shoulder_clip_landmarks[0], shoulder_clip_landmarks[1]))
                if current_clip_range is None and slope > shoulder_threshold:
                    current_clip_range = {'start_frame': current_frame, 'slope': slope}
                    shoulder_slope_triggered = True
                elif current_clip_range is not None and slope < shoulder_threshold:
                    current_clip_range['end_frame'] = current_frame
                    shoulder_clip_ranges.append(current_clip_range)
                    current_clip_range = None
                    shoulder_slope_triggered = False
                    csv_generated = False
                if shoulder_slope_triggered and not csv_generated:
                    generate_csv_file3(clip_name, shoulder_slope_triggered, output_folder)
                    csv_generated = True
            current_frame += 1
        else:
            break

    if current_clip_range is not None:
        current_clip_range['end_frame'] = current_frame
        shoulder_clip_ranges.append(current_clip_range)

    cap.release()
    cv2.destroyAllWindows()
    return shoulder_clip_ranges

def process_video(input_path, clip_name, output_folder):
    knee_clip_landmarks = [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE]
    hip_clip_landmarks = [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]
    shoulder_clip_landmarks = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]

    knee_clip_ranges = find_clip_ranges(input_path, knee_clip_landmarks, clip_name, output_folder)
    hip_clip_ranges = find_clip_ranges2(input_path, hip_clip_landmarks, clip_name, output_folder)
    shoulder_clip_ranges = find_clip_ranges3(input_path, shoulder_clip_landmarks, clip_name, output_folder)

    output_video_path = os.path.join(output_folder, f"{clip_name}_processed.mp4")
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        for knee_clip_range in knee_clip_ranges:
            if knee_clip_range['start_frame'] <= current_frame <= knee_clip_range['end_frame']:
                frame = adjust_color(frame)
        for hip_clip_range in hip_clip_ranges:
            if hip_clip_range['start_frame'] <= current_frame <= hip_clip_range['end_frame']:
                frame = adjust_color(frame)
        for shoulder_clip_range in shoulder_clip_ranges:
            if shoulder_clip_range['start_frame'] <= current_frame <= shoulder_clip_range['end_frame']:
                frame = adjust_color(frame)
        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
