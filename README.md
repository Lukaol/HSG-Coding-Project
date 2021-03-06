# Face Mask Detector
Please read this README before running the code. You will find all information in order to run the code below.

**About**

This is a machine learning project for Python of Matthias Brüderlin, Daan Friese, Oleg Lukanin, Celia Vetter and Joel Weibel. The project was created for the course Skills: Programming with Advanced Computer Languages at the University of St.Gallen. The goal of the project was to create a real time face mask detector using Python, Keras, OpenCV and MobileNet. The program can recognize if somebody is wearing a face mask, either on a live video or on a recorded video and allows the user to save the video with the corresponding results.

**Prerequisites**

This program is written for Python. The following versions have been used to run it:
1. Python 3.8 from Anaconda (64-bit)
2. Pycharm Community Edition (2020.2.3)
3. Microsoft Visual C++ 2015-2019 Redistributable


Before running the code, please do the following:
1.	Download the folders "dataset" and "face_detector" as well as the document "requirements.txt" and safe it in a folder (e.g. mask detector program) on your computer
2.	Install the necessary packages: 

     i. Open anaconda prompt 
  
     ii.	Enter "cd" and copy paste your working directory to the document "requirements.txt" (e.g. cd C:\Users\XY\mask detector program”) and press enter 
  
     iii.	Then, type "pip install -r requirements.txt" in order to install all the dependencies and required libraries defined in the document
  
3.	Open Pycharm and change your working directory to the folder created in step 1 (click on the right top on your file name -> choose “Edit Configurations…" -> change your script path to the corresponding folder)
4.	Choose your python interpreter (click on the right top on your file name -> choose “Edit Configurations…"-> choose anaconda as your python interpreter)
5.	Change your direction in the file "train_mask_detector.py" to the corresponding folder with the datasets (line 30) (e.g. DIRECTORY = r"C:\Users\XY\mask detector program\dataset”) If you open it as a project and all project files are in the same folder, this change is not necessary
6.	Run the code "train_mask_detector.py" or "train_mask_detector_short_training.py" in order to train the model
7.	Run the code "detect_mask_video.py" in order to apply the trained model

**Description**

The mask detector consists of two files. The first one trains the model and the second one analyses the live video or captured video by means of the trained model.

1. a) train_mask_detector.py
This code trains the mask detector model which will be used later to analyze a live video or captured video. First, it loads the dataset which consists of two folders with pictures: one with people wearing a mask and one with people without a mask. Afterwards, it will train the face mask detector by means of the labelled pictures. The program saves our detector as "mask_detector.model" and creates a plot with the losses and the accuracy of our model.
To prevent overfitting to the training set, we introduced an early stop. When the validation loss does not decrease for three epochs, the model stops training and reverts to using the best weights.

   b) train_mask_detector_short_training.py
This is a shorter version of the code that trains the mask detector model. It can be used instead of the train_mask_detector.py in order to save time.  It works the same way as described in 1 a). The time for training was made shorter by introducing an Early Stop that terminates the training once the validation accuracy stops increasing meaningfully. 

2.	detect_mask_video.py
In the second code, we first implement a face detection. The model created in the other file is then loaded into our second program. Then, you can apply the trained program in the following two ways: it will give you the choice to either activate your camera and check if you are wearing a face mask or upload a video to which the mechanism is applied. It furthermore indicates the percentage of certainty whether it detects a face mask or not. At the end, the user can choose if he wants to save the video or not.

**Results**

Our model gave 99% accuracy for face mask detection after training:

![alt text](https://github.com/Lukaol/HSG-Coding-Project/blob/main/Accuracyfacemaskdetector.png)

We got the following accuracy/loss training curve plot:

![alt text](https://github.com/Lukaol/HSG-Coding-Project/blob/main/Accuracylossplot.png)

This is how it looks on your computer if you successfully ran the code:

![alt text](https://github.com/Lukaol/HSG-Coding-Project/blob/main/TheGIF.gif)

**Sources**

We followed a tutorial by Balaji Srinivas, available at https://www.youtube.com/watch?v=Ax6P93r32KU&t. The dataset has been downloaded from his GitHUb repository - https://github.com/balajisrinivas/Face-Mask-Detection. We have tweaked the learning process of the model and the program has also been enhanced with different functionalities. You are able to choose between using the local webcam or uploading a video that is saved on your computer. It is possible to save the output as a .avi video with your chosen filename and location.

**Disclaimer**

The program worked fine with PyCharm 2020 and Python 3.8 (64-bit) in December 2020. Running the code with other versions of Python or programs may cause errors. For example, TensorFlow is only working for Python 3.5-3.8 (64-bit). Moreover, we used TensorFlow 2.3.0 which is compatible with Python 3.9. For other Python versions you might change the TensorFlow Version. Further, the website links and the available data on these websites may have changed in the meantime.

