# Facial Detection and Recognition Programs
By Akash Ravandhu
### This project contains four sub-projects:
- FacialDetectionPhoto: Face detection with a single photo.
	- Input: Photo
	- Output: Photo + box overlay if face(s) identified
- **FacialDetectionVideo**: Face detection with a live video feed from the user's device.
	- Input: Live Camera View
	- Output: Live Camera View + box overlay on face(s)
		- Feature: Live FPS (frames per second) counter to track model performance
- FacialRecognitionPhoto: Face detection & Face Recognition comparing the faces in two images.
	- Input: 2 images with a single face in each
	- Output: "Same Person" or "Not same person" depending on whether the same face is identified in both photos.
- **FacialRecognitionVideo**: Face detection & Face Recognition in a live video feed.
	- Input: Initialization step in which a single photo is taken of the desired "verified" face.
	- Output: Live Camera View + box overlay on face(s) + Identified/Unidentified label
		- If the face matches with the "verified" face, it is labelled as "Identified".
		- If the face does not match with the "verified" face, it is labelled as "Unidentified".

> Note: *FacialDetectionPhoto* and *FacialRecognitionPhoto* were used as steps to build the video complement to each. The "final products" of this project are ***FacialDetectionVideo*** and ***FacialRecognitionVideo***.

### Frameworks/Packages Used:
- PyTorch
	- Facenet-PyTorch (for pretrained MTCNN and InceptionResnetV1 models)
	- Torchvision module (for `transforms`: Image -> PyTorch tensor)
- Image/Video Capture, Processing, and Display
	- OpenCV
	- PIL/Pillow
