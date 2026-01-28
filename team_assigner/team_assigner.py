from sklearn.cluster import KMeans
import supervision as sv
from tqdm import tqdm
import torch
from inference import get_model

from sports.common.team import TeamClassifier
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class TeamAssigner:
    def __init__(self, path=None, model_id=None, api_key=None):
        self.team_colors = {}
        self.player_team_dict = {}
        self.path = path
        self.model = get_model(model_id=model_id, api_key=api_key)
        self.team_classifier = TeamClassifier(device=DEVICE)

    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1).fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame, player_detections):
        
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10).fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def colect_player_team_data(self, player_detections):
        frame_generator = sv.get_video_frames_generator(source_path=self.path, stride=30)
    
        crops = []
        # Lặp qua video để lấy mẫu áo cầu thủ
        for frame in tqdm(frame_generator, desc='Collecting crops'):
            result = self.model.infer(frame, confidence=0.3)[0]
            detections = sv.Detections.from_inference(result)
            
            # Chỉ lấy crop của cầu thủ (PLAYER_ID)
            players_detections = detections[detections.class_id == 2]
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            crops += players_crops
        return crops
    def fit_team_classifier(self, crops):
        self.team_classifier.fit(crops)


    def get_player_team(self, frame, player_bbox):
        
        player_crop = sv.crop_image(frame, player_bbox)

        team_id = self.team_classifier.predict([player_crop])[0] 

        return team_id