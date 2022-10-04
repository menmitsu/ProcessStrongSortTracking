import os
import sys
import argparse
import string
import cv2
import mediapipe as mp
import numpy as np
import json 
import math
from PIL import ImageFont, ImageDraw, Image  


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


dataset_folder="20220914_134240"
file_path="Datasets/"+dataset_folder+"/labels/annotations.txt"
input_video="Datasets/"+dataset_folder+"/input_video/video.mp4"
sortedFoldersBasePath="Sorted"


cap = cv2.VideoCapture(input_video)
num_frames = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))

frames_Dict={}
angles_Dict={}



def calculate_angle(a, b, c):
	a = np.array(a)  # First
	b = np.array(b)  # Mid
	c = np.array(c)  # End

	radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
	angle = np.abs(radians * 180.0 / np.pi)

	if angle > 180.0:
		angle = 360 - angle

	return angle

# Currently only angle between hip->shoulder->elbow is calculated. 
# ToDo:Extend this for other shoulder 

def cal_head_palm_dist(landmarks,width,height):
		face = [width*landmarks[mp_pose.PoseLandmark.NOSE.value].x,
					   height*landmarks[mp_pose.PoseLandmark.NOSE.value].y]

		wrist_right = [width*landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
					   height*landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
		wrist_left = [width*landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
					  height*landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]



		d1 = math.dist(face, wrist_left)
		d2 = math.dist(face, wrist_right)
		print(d1)

		# What if one hand is on the side and other is holding bag
		dist = min(d1, d2)
		return dist





def calculate_joint_angles(landmarks,img,frameNum,tracking_id):

	# Get coordinates of Right shoulder
	right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
				landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
	right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
			 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
	right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
			 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

	  # Get coordinates of Right shoulder
	left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
				landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
	left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
			 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
	left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
			 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

	left_knee=[landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
				landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
	right_knee=[landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
				landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]


	left_wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
				landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
	right_wrist=[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
				landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

	
	current_frame_all_angles_list={}



	height, width = img.shape[:2]
	img2 = np.zeros((int(height),int(width),3), dtype=np.uint8)


	print("Image Width and height"+str(height))
	
	# height=height*landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
	# height=height*right_shoulder[1]
	# # width=width*landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
	# width=width*right_shoulder[0]   


	rhse_angle_text_location=np.array([int(width*right_shoulder[0]),int(height*right_shoulder[1])])
	lhse_angle_text_location=np.array([int(width*left_shoulder[0]),int(height*left_shoulder[1])])

	rsew_angle_text_location=np.array([int(width*right_elbow[0]),int(height*right_elbow[1])])
	lsew_angle_text_location=np.array([int(width*left_elbow[0]),int(height*left_elbow[1])])

	rshk_angle_text_location=np.array([int(width*right_hip[0]),int(height*right_hip[1])])
	lshk_angle_text_location=np.array([int(width*left_hip[0]),int(height*left_hip[1])])


	image_dimensions=np.array([width,height])


	lhse_angle_text_location=np.subtract(image_dimensions,lhse_angle_text_location)
	rhse_angle_text_location=np.subtract(image_dimensions,rhse_angle_text_location)
	
	rsew_angle_text_location=np.subtract(image_dimensions,rsew_angle_text_location)
	lsew_angle_text_location=np.subtract(image_dimensions,lsew_angle_text_location)

	rshk_angle_text_location=np.subtract(image_dimensions,rshk_angle_text_location)
	lshk_angle_text_location=np.subtract(image_dimensions,lshk_angle_text_location)

	print(lhse_angle_text_location)
  
	
	rhse_angle = calculate_angle(right_hip,right_shoulder, right_elbow)
	lhse_angle = calculate_angle(left_hip,left_shoulder, left_elbow)

	rsew_angle = calculate_angle(right_shoulder,right_elbow, right_wrist)
	lsew_angle = calculate_angle(left_shoulder,left_elbow, left_wrist)

	rshk_angle = calculate_angle(right_shoulder,right_hip,right_knee)
	lshk_angle = calculate_angle(left_shoulder,left_hip, left_knee)

	face_to_hand_dist=cal_head_palm_dist(landmarks,width,height)

   
	print("Face to hand dist")
	print(face_to_hand_dist)

	if tracking_id not in angles_Dict.keys():
		angles_Dict[tracking_id]={}

	current_frame_all_angles_list["RH,RS,RE"]=rhse_angle
	current_frame_all_angles_list["LH,LS,LE"]=lhse_angle

	current_frame_all_angles_list["RS,RE,RW"]=rsew_angle
	current_frame_all_angles_list["LS,LE,LW"]=lsew_angle

	current_frame_all_angles_list["RS,RH,RK"]=rshk_angle
	current_frame_all_angles_list["LS,LH,LK"]=lshk_angle

	current_frame_all_angles_list["FTH"]=face_to_hand_dist

	angles_Dict[tracking_id][frameNum]=current_frame_all_angles_list
	

	#  Since most videos are flipped, this is a temporary hack to display text that will be flipped-vertically.
	
	
	# cv2.putText(img2, str(round(rhse_angle,2)), rhse_angle_text_location, cv2.FONT_HERSHEY_SIMPLEX, .5, 255, 2, cv2.LINE_AA)
	# cv2.putText(img2, str(round(lhse_angle,2)), lhse_angle_text_location, cv2.FONT_HERSHEY_SIMPLEX, .5, 255, 2)
	# cv2.putText(img2, str(round(rsew_angle,2)), rsew_angle_text_location, cv2.FONT_HERSHEY_SIMPLEX, .5, 255, 2)    
	# cv2.putText(img2, str(round(lsew_angle,2)), lsew_angle_text_location, cv2.FONT_HERSHEY_SIMPLEX, .5, 255, 2)
	# cv2.putText(img2, str(round(rshk_angle,2)), rshk_angle_text_location, cv2.FONT_HERSHEY_SIMPLEX, .5, 255, 2, cv2.LINE_AA)    
	# cv2.putText(img2, str(round(lshk_angle,2)), lshk_angle_text_location, cv2.FONT_HERSHEY_SIMPLEX, .5, 255, 2)

	# Using truetype font for crisper text

	pil_im = Image.fromarray(img2)  

	draw = ImageDraw.Draw(pil_im)
	font_size=20
	font = ImageFont.truetype("futura.ttf", font_size)

	draw.text(rhse_angle_text_location, str(round(rhse_angle,2)), font=font,fill="#FFFFFF")
	draw.text(lhse_angle_text_location, str(round(lhse_angle,2)), font=font,fill="#FFFFFF")
	draw.text(rsew_angle_text_location, str(round(rsew_angle,2)), font=font,fill="#FFFFFF")
	draw.text(lsew_angle_text_location, str(round(lsew_angle,2)), font=font,fill="#FFFFFF")
	draw.text(rshk_angle_text_location, str(round(rshk_angle,2)), font=font,fill="#FFFFFF")
	draw.text(lshk_angle_text_location, str(round(lshk_angle,2)), font=font,fill="#FFFFFF")

	cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
	cv2.imshow("imagettf",cv2_im_processed)
	



	# M = cv2.getRotationMatrix2D(lhse_angle_text_location, 180, 1)
	# out = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))

	# img=cv2.add(cv2.merge((out,out,out)),img)
	img3=cv2.merge((img2,img2,img2))
	  

	# Visualize angle
	# cv2.putText(img, str(angle), (int(width),int(height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

	return cv2_im_processed


# Read annotations file from dataset_folder.
# frames_Dict is a nested dictionary with frameNumber as the key.
# At each entry, there is another dictionary which stores information of the annotations coordinates using the strong-sort-id as key

with open(file_path) as file:
	lines = file.readlines()
	line_num = 0

	for line in lines:
		
		line_num += 1
		print(f"line {line_num}: {line}")
		split_values=line.split()

		if not split_values[0] in frames_Dict.keys():
			frames_Dict[split_values[0]]={}   

		# print(split_values)

		# frames_Dict[fameNumber][tracking_id_number]=(x,y,width,height)
		frames_Dict[split_values[0]][split_values[2]]=[int(split_values[3]),int(split_values[4]),int(split_values[5]),int(split_values[6])]

for i in frames_Dict:
	print(i, '->', frames_Dict[i])

print(frames_Dict.keys())

frameCount=1
if not os.path.exists(sortedFoldersBasePath):
	os.makedirs(sortedFoldersBasePath)

cv2.namedWindow("Input", cv2.WINDOW_AUTOSIZE)

while cap.isOpened():
	ret, img = cap.read()
	if not ret:
		break
	
	print("Processing FrameNumber:"+str(frameCount))

	if str(frameCount) in frames_Dict.keys():
		for id in frames_Dict[str(frameCount)]:
			print(frames_Dict[str(frameCount)][id])

			if not os.path.exists(sortedFoldersBasePath+"/"+id):
				os.makedirs(sortedFoldersBasePath+"/"+id)


			starting_pt=(frames_Dict[str(frameCount)][id][0],frames_Dict[str(frameCount)][id][1])
			ending_pt=(frames_Dict[str(frameCount)][id][0]+frames_Dict[str(frameCount)][id][2],frames_Dict[str(frameCount)][id][1]+frames_Dict[str(frameCount)][id][3])
	
			
			ROI = img[starting_pt[1]:ending_pt[1], starting_pt[0]:ending_pt[0]]
			img_with_joint_angles=None
			
			with mp_pose.Pose(
				static_image_mode=True,
				model_complexity=2,
				enable_segmentation=True,
				min_detection_confidence=0.5) as pose:

				results = pose.process(cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB))
				if not results.pose_landmarks:
					continue
				else:
					img_with_joint_angles=calculate_joint_angles(results.pose_landmarks.landmark,ROI,frameCount,id)

				
				# mp_drawing.draw_landmarks(img_with_joint_angles, results.pose_landmarks, mp_pose. POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
				mp_drawing.draw_landmarks(ROI, results.pose_landmarks, mp_pose. POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
					

				cv2.imwrite('{}.png'.format(sortedFoldersBasePath+"/"+id+"/"+str(frameCount)), cv2.add(img_with_joint_angles,cv2.flip(ROI,-1)))
				cv2.rectangle(img, starting_pt, ending_pt, (255,255,255), 1)
	
	cv2.imshow("Input",cv2.flip(img,-1))
	frameCount=frameCount+1


	if cv2.waitKey(1) & 0xFF == 27:
	  break
	

print(angles_Dict)
# json_object = json.dumps(dictionary, indent = 4) 
angles_file_path=sortedFoldersBasePath+"/angles.json"
with open(angles_file_path, "w") as outfile:
	json.dump(angles_Dict, outfile,indent = 4) 
# print(angles_Dict)	
# cv2.waitKey(0)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     