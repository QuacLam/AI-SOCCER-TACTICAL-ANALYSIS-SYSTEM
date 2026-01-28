from inference import get_model
from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import supervision as sv

def main():
    # 1. Cấu hình & Đọc Video
    VIDEO_PATH = "./input_videos/08fd33_4.mp4"
    ROBOFLOW_API_KEY = 'vHi8SqZ8FtEnZeJavRY6'
    PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"

    video_frames = read_video(VIDEO_PATH)
    # 2. Khởi tạo Tracker (Cập nhật tham số cho Roboflow API)
    tracker = Tracker(
        model_id=PLAYER_DETECTION_MODEL_ID, 
        api_key=ROBOFLOW_API_KEY
    )

    # Lấy Tracks
    # read_from_stub=False để chạy lại detection mới nhất
    tracks = tracker.get_object_tracks(
        video_frames, 
        read_from_stub=False, 
        stub_path="stubs/track_stub.pkl"
    )
    
    # Bổ sung vị trí và nội suy bóng
    tracker.add_position_to_tracks(tracks)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # 3. Phân loại Team
    team_assigner = TeamAssigner(
        path=VIDEO_PATH,
        model_id=PLAYER_DETECTION_MODEL_ID,
        api_key=ROBOFLOW_API_KEY
        )
    
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])
    crops = team_assigner.colect_player_team_data(tracks["players"][0])
    team_assigner.fit_team_classifier(crops)

    # Gán team cho từng cầu thủ trong từng frame
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            # Gọi hàm get_player_team (Sử dụng model đã train ở trên)
            
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"]
            )
            
            # Lưu kết quả
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team + 1]

    
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]

       
        if np.isnan(ball_bbox).any():
            assigned_player = -1
        else:
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        # -------------------------

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(
                tracks["players"][frame_num][assigned_player]["team"]
            )
        else:
            # Nếu không ai giữ bóng, lấy trạng thái của frame trước
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(1) 
                
    team_ball_control = np.array(team_ball_control)

    # 5. Vẽ và Xuất Video
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control
    )

    # Debug: In thử dữ liệu frame đầu tiên để kiểm tra
    print("Debug Frame 0 Data:")
    if 0 in tracks["players"]:
        for player_id, data in tracks["players"][0].items():
            print(f"Player ID: {player_id} - Data: {data}")

    save_video(output_video_frames, "./output_videos/output_video.avi")

if __name__ == "__main__":
    main()