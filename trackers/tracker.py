from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
from inference import get_model

class Tracker:
    def __init__(self, model_id, api_key):
        self.model = get_model(model_id=model_id, api_key=api_key)

        self.tracker = sv.ByteTrack()

        self.BALL_ID = 0
        self.GOALKEEPER_ID = 1
        self.PLAYER_ID = 2
        self.REFEREE_ID = 3
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
    
    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        # 1. Chuyển đổi dữ liệu thô: Thay thế [] bằng [NaN, NaN, NaN, NaN]
        ball_positions_fixed = []
        for x in ball_positions:
            # Lấy bbox từ dict {1: {'bbox': ...}}
            bbox = x.get(0, {}).get('bbox', [])
            
            # Nếu không có dữ liệu, điền NaN để giữ chỗ (tránh lỗi lệch cột)
            if len(bbox) == 0:
                ball_positions_fixed.append([np.nan, np.nan, np.nan, np.nan])
            else:
                ball_positions_fixed.append(bbox)
        
        # 2. Tạo DataFrame (Lúc này đã đủ 4 cột, không còn lỗi ValueError)
        df_ball_positions = pd.DataFrame(ball_positions_fixed, columns=['x1','y1','x2','y2'])

        # 3. XỬ LÝ ĐIỀN DỮ LIỆU THIẾU
        # Cách 1 (Khuyên dùng): Nội suy tuyến tính -> Bóng di chuyển mượt
        df_ball_positions = df_ball_positions.interpolate()
        
        # Cách 2 (Ý của bạn): Dùng giá trị trước đó (Forward Fill)
        # Nếu bạn thực sự thích cách này thì bỏ comment dòng dưới và đóng dòng interpolate trên
        # df_ball_positions = df_ball_positions.ffill()

        # Dùng bfill để xử lý trường hợp mất dấu ngay frame đầu tiên
        df_ball_positions = df_ball_positions.bfill()

        # 4. Chuyển ngược lại format list dictionary
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        all_detections = []
        for frame_num, frame in enumerate(frames):
            result = self.model.infer(frame, confidence=0.2)[0]
            detections = sv.Detections.from_inference(result)
            all_detections.append(detections)
        return all_detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            print(f"Loaded tracks from stub: {stub_path}")
            return tracks
        detections = self.detect_frames(frames)
        tracks = {'players':[], 'ball':[], 'referees':[], 'goalkeepers':[]}
        for frame_num, detection in enumerate(detections):
            # Lọc detections theo class_id
            ball_detections = detection[detection.class_id == self.BALL_ID]

            all_detections = detection[detection.class_id != self.BALL_ID]
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
            all_detections = self.tracker.update_with_detections(detections=all_detections)

            players_detections = all_detections[all_detections.class_id == self.PLAYER_ID]
            referees_detections = all_detections[all_detections.class_id == self.REFEREE_ID]
            goalkeepers_detections = all_detections[all_detections.class_id == self.GOALKEEPER_ID]

            tracks['players'].append({})
            tracks['goalkeepers'].append({})
            tracks['ball'].append({})
            tracks['referees'].append({})
            
            for frame_detection in ball_detections:
                tracks['ball'][frame_num][0] = {'bbox': frame_detection[0]}
            for frame_detection in players_detections:
                tracks['players'][frame_num][frame_detection[4]] = {'bbox': frame_detection[0]}
            for frame_detection in referees_detections:
                tracks['referees'][frame_num][frame_detection[4]] = {'bbox': frame_detection[0]}
            for frame_detection in goalkeepers_detections:
                tracks['goalkeepers'][frame_num][frame_detection[4]] = {'bbox': frame_detection[0]}
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # 1. Vẽ hình chữ nhật nền bán trong suốt
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 2. Lấy dữ liệu kiểm soát bóng tính đến frame hiện tại
        team_ball_control_till_frame = team_ball_control[:frame_num+1]

        # 3. Đếm số lần mỗi đội kiểm soát bóng
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]

        # --- PHẦN ĐÃ SỬA LỖI ZERO DIVISION ---
        total_control_frames = team_1_num_frames + team_2_num_frames

        if total_control_frames == 0:
            # Nếu chưa ai giữ bóng, gán bằng 0 để tránh lỗi chia cho 0
            team_1 = 0
            team_2 = 0
        else:
            # Tính tỷ lệ phần trăm bình thường
            team_1 = team_1_num_frames / total_control_frames
            team_2 = team_2_num_frames / total_control_frames
        # -------------------------------------

        # 4. Viết text lên màn hình
        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                ball_bbox = ball["bbox"]

                
                if np.isnan(ball_bbox).any():
                    continue

                frame = self.draw_traingle(frame, ball_bbox, (0, 255, 0))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames