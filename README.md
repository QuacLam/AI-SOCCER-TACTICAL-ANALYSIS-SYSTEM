**Football Match Analysis using Computer Vision**


**Overview**

This project develops a computer vision system for automatic football match analysis using deep learning and video processing techniques.
The system detects and tracks players, referees, and the ball from broadcast football videos and generates useful match statistics.
Unlike traditional sports analytics systems that rely on GPS sensors or specialized cameras, this approach works directly from standard match videos, making it more accessible and scalable.


**Key Features**

The system provides several automated football analytics features:
Object Detection – Detect players, referees, goalkeepers, and the ball using YOLOv8
Multi-Object Tracking – Track each object across video frames using ByteTrack
Player Speed Estimation – Calculate player movement speed
Distance Traveled – Estimate total distance covered by players
Ball Possession Analysis – Measure ball possession time for each team
Mini-Map Visualization – Display player positions on a tactical field map
These features help extract tactical and performance insights directly from match footage.


**System Pipeline**

Input Video
     ↓
Frame Extraction (OpenCV)
     ↓
Object Detection (YOLOv8)
     ↓
Object Tracking (ByteTrack)
     ↓
Analytics Module
     ↓
Speed / Distance / Possession / Mini-Map


**Dataset**

The model was trained using a football players detection dataset containing annotated images with the following classes:
Player
Goalkeeper
Referee
Ball
Images were resized to 640×640 and converted to YOLO format for training.


**Technologies Used**

Python
YOLOv8
PyTorch
OpenCV
ByteTrack
Roboflow


**Results**

The trained detection model achieved:
Metric	Score
Precision	0.923
Recall	0.763
mAP@0.5	0.833


**Applications**

This system can support:
Sports performance analysis
Tactical analysis for teams
Broadcast analytics
Automated football statistics generation


**Future Improvements**

Improve ball detection in crowded scenes
Real-time match analysis
Advanced tactical pattern recognition
