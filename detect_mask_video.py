# importing the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import os
import tkinter.filedialog

def detect_and_predict_mask(frame, faceNet, maskNet):
	# Finding the dimensions of the frame
	(h, w) = frame.shape[:2]
	# Making a blob from the frame
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# Passing the blob through the neural network and obtaining detected faces from it
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# Initializing the list of faces, their corresponding locations,
	# and the list of predictions from the face mask neural network
	faces = []
	locs = []
	preds = []

	# Looping over the face detections
	for i in range(0, detections.shape[2]):
		# Taking the probability associated with the detection
		confidence = detections[0, 0, i, 2]

		# Discarding weak detections by setting a minimum confidence/probability
		if confidence > 0.5:
			# Computing coordinates of the face box
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Making sure that the face box is inside the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# Processing the face image
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# Adding the faces and coordinates to the lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# Making sure predictions are made only if at least one face is detected
	if len(faces) > 0:
		# For multiple faces detections will be done at the same time instead of one by one
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# Returning a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# Loading the face detector model
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Loading our trained face mask detector model
mask_detector_model = r"mask_detector.model"
maskNet = load_model(mask_detector_model)

# Asking whether the user wants to use a live webcam
# or open an existing video

userOptions = ['webcam', 'video']
while True:
	userChoice = input('Would you like to use the webcam or open an existing video? Type "webcam" or "video". ').lower()
	if userChoice in userOptions:
		break
	else:
		print('Please, type "webcam" or "video"')

if userChoice == 'video':
	# Asking the user for the file
	print('Please, pick the file.')
	videoDirectory = tkinter.filedialog.askopenfilename()
	vs = FileVideoStream(videoDirectory).start()
else:
	# initializing the video stream
	print("[INFO] Starting video stream...")
	vs = VideoStream(src=0, resolution=(1280, 720)).start()


# Asking whether the user wants to save the output video
userSaveChoiceOptions = ['Y', 'N']
while True:
	userSaveChoice = input('Would you like to save the output video? (Y/N) ')
	if userSaveChoice in ['Yes', 'yes', 'y', 'YES']:
		userSaveChoice = 'Y'
	elif userSaveChoice in ['No', 'no', 'n', 'NO']:
		userSaveChoice = 'N'
	if userSaveChoice.upper() in userSaveChoiceOptions:
		break
	else:
		print('Please, type "Y" or "N".')

# Asking where the user wants to save the video
# and with what name
if userSaveChoice == 'Y':
	print('Where would you like to save the output video?')
	saveDirectory = tkinter.filedialog.askdirectory()
	print(saveDirectory)
	saveName = input('What would you like to name the output video? ')
	joiner = (saveDirectory, '/', saveName, '.avi')
	# Output directory
	saveOutputDirectory = ''.join(joiner)

# Initializing some statistics
totalVideoTime = 0
maskWearing = 0
listFrames = []
# Start FPS Counter
fps = FPS().start()

# Looping over the frames from the webcam stream or video
while True:
	# Taking the frame from video feed
	frame = vs.read()

	# Stopping the loop if the user used a video file and the video finished
	if frame is None:
		break

	# Resizing the frame
	frame = imutils.resize(frame, width=1000)

	# Detecting faces in the frame and determining whether they are wearing a mask
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# Looping over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# Unpacking the bounding box and mask predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		totalVideoTime += 1
		if mask > withoutMask:
			maskWearing += 1

		# Picking labels and colors according to results
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# Adding probability to the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# Displaying the labels and box on the output frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	listFrames.append(frame)

	# Telling the user how to stop the webcam interface
	if userChoice == 'webcam':
		cv2.putText(frame, 'Press "Q" to stop the webcam', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,0), 2)

	# Showing the output frame
	cv2.imshow("Detecting Face Mask...", frame)
	key = cv2.waitKey(1) & 0xFF
	fps.update()

	# If the `q` key was pressed, breaking from the loop
	if key == ord("q"):
		break

# Stop FPS counter
fps.stop()

# Saving video
if userSaveChoice == 'Y':
	size = listFrames[1].shape[1], listFrames[1].shape[0]
	output = cv2.VideoWriter(saveOutputDirectory, cv2.VideoWriter_fourcc('M','J','P','G'), fps.fps(), size)
	for i in listFrames:
		output.write(i)
	output.release()

cv2.destroyAllWindows()
vs.stop()

# Showing the user how long the mask was worn (only works with one person)
print('The mask was probably worn', maskWearing/totalVideoTime * 100, '% of the time.')
