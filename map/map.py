from inference import get_model
from sports.common.soccer import (
    draw_pitch,
    draw_points_on_pitch,
    draw_pitch_voronoi_diagram
)
from sports.common.config_soccer import SoccerPitchConfiguration
import supervision as sv
import numpy as np
from sports.common.view_transformer import ViewTransformer
class SoccerFieldMapper:
    def __init__(self, model_id=None, api_key=None, ball_detections=None, players_detections=None, referees_detections=None):
        self.pitch_configuration = SoccerPitchConfiguration()
        self.model = get_model(model_id=model_id, api_key=api_key)

    def map_field(self, frame):
        result = self.model.infer(frame, confidence=0.3)[0]
        key_points = sv.KeyPoints.from_inference(result)
        filter = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter]
        pitch_reference_points = np.array(self.pitch_configuration.vertices)[filter]
        transformer = ViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points
        )
        frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) # type: ignore
        pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

        players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) # type: ignore
        pitch_players_xy = transformer.transform_points(points=players_xy)

        referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) # type: ignore
        pitch_referees_xy = transformer.transform_points(points=referees_xy)
        annotated_frame = draw_pitch(self.pitch_configuration)
        annotated_frame = draw_points_on_pitch(
            config=self.pitch_configuration,
            xy=pitch_ball_xy,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=annotated_frame)
        annotated_frame = draw_points_on_pitch(
            config=self.pitch_configuration,
            xy=pitch_players_xy[players_detections.class_id == 1], # type: ignore
            face_color=sv.Color.from_hex('00BFFF'),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=annotated_frame)
        annotated_frame = draw_points_on_pitch(
            config=self.pitch_configuration,
            xy=pitch_players_xy[players_detections.class_id == 2], # type: ignore
            face_color=sv.Color.from_hex('FF1493'),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=annotated_frame)
        annotated_frame = draw_points_on_pitch(
            config=self.pitch_configuration,
            xy=pitch_referees_xy,
            face_color=sv.Color.from_hex('FFD700'),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=annotated_frame)