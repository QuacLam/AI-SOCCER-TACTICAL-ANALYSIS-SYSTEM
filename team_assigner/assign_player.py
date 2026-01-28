import torch
import numpy as np
import supervision as sv
from tqdm import tqdm
from inference import get_model
from transformers import AutoProcessor, SiglipVisionModel
from sports.common.team import TeamClassifier

# --- CẤU HÌNH (CONSTANTS) ---
SOURCE_VIDEO_PATH = "input_videos/121364_0.mp4"
TARGET_VIDEO_PATH = "output_video.mp4"
ROBOFLOW_API_KEY = 'vHi8SqZ8FtEnZeJavRY6' # Lưu ý bảo mật API Key của bạn
PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"

# Định nghĩa ID của các Class
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# --- KHỞI TẠO MODELS ---
# 1. Detection Model (Roboflow)
print("Loading Detection Model...")
PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

# 2. Embedding Model (cho Team Classifier) - Phần này nằm trong logic của TeamClassifier nhưng cần khai báo device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- HELPER FUNCTIONS ---
def resolve_goalkeepers_team_id(
    players: sv.Detections,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Gán team cho thủ môn dựa trên khoảng cách tới centroid của từng đội
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    
    # Tính tâm (centroid) của đội 0 và đội 1
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)

def main():
    # --- GIAI ĐOẠN 1: TRAINING TEAM CLASSIFIER ---
    print("\n--- GIAI ĐOẠN 1: Thu thập dữ liệu để phân loại đội ---")
    
    # Stride = 30 nghĩa là cứ 30 frames lấy 1 frame để train cho nhanh
    frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH, stride=30)
    
    crops = []
    # Lặp qua video để lấy mẫu áo cầu thủ
    for frame in tqdm(frame_generator, desc='Collecting crops'):
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        
        # Chỉ lấy crop của cầu thủ (PLAYER_ID)
        players_detections = detections[detections.class_id == PLAYER_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        crops += players_crops

    # Train classifier
    print(f"Đang train classifier với {len(crops)} mẫu áo...")
    team_classifier = TeamClassifier(device=DEVICE)
    team_classifier.fit(crops)
    print("Training hoàn tất!")

    # --- GIAI ĐOẠN 2: VIDEO INFERENCE & TRACKING ---
    print("\n--- GIAI ĐOẠN 2: Xử lý toàn bộ video ---")

    # Khởi tạo các công cụ vẽ (Annotators)
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=25,
        height=21,
        outline_thickness=1
    )

    # Khởi tạo Tracker
    tracker = sv.ByteTrack()
    tracker.reset()

    # Khởi tạo Video I/O
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH) # Không dùng stride ở đây

    # Mở VideoSink để ghi file
    with sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames, desc="Processing Video"):
            
            # 1. Detect
            result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.2)[0]
            detections = sv.Detections.from_inference(result)

            # 2. Tách Ball
            ball_detections = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            # 3. Tracking các đối tượng khác
            all_detections = detections[detections.class_id != BALL_ID]
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
            all_detections = tracker.update_with_detections(detections=all_detections)

            # 4. Phân nhóm
            goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
            players_detections = all_detections[all_detections.class_id == PLAYER_ID]
            referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

            # 5. Predict Team (Cầu thủ)
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            # Predict team cho cầu thủ
            players_detections.class_id = team_classifier.predict(players_crops)

            # 6. Resolve Team (Thủ môn)
            # Nếu có cả cầu thủ và thủ môn thì mới resolve được
            if len(players_detections) > 0 and len(goalkeepers_detections) > 0:
                goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
                    players_detections, goalkeepers_detections)
            
            # 7. Xử lý Trọng tài (tránh trùng màu team 0/1)
            referees_detections.class_id -= 1

            # 8. Merge lại để vẽ
            all_detections = sv.Detections.merge([
                players_detections, goalkeepers_detections, referees_detections])

            # Tạo labels (#ID)
            labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]

            # Fix lỗi visualize màu (ép kiểu về int)
            all_detections.class_id = all_detections.class_id.astype(int)

            # 9. Vẽ lên frame
            annotated_frame = frame.copy()
            
            # Vẽ vòng tròn dưới chân cầu thủ/trọng tài
            annotated_frame = ellipse_annotator.annotate(
                scene=annotated_frame,
                detections=all_detections)
            
            # Vẽ nhãn ID
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=all_detections,
                labels=labels)
            
            # Vẽ tam giác trên đầu bóng
            annotated_frame = triangle_annotator.annotate(
                scene=annotated_frame,
                detections=ball_detections)

            # 10. Lưu frame vào video
            sink.write_frame(annotated_frame)

    print(f"Xong! Video đã được lưu tại: {TARGET_VIDEO_PATH}")

if __name__ == "__main__":
    main()