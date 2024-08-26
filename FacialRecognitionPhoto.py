import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image


def get_embeddings(param_image):
    # Convert OpenCV image to PIL image
    color_converted = cv2.cvtColor(param_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(color_converted)

    # Extract face embeddings
    aligned = mtcnn(img).unsqueeze(0)
    return resnet(aligned).detach()

# Initialize MTCNN for facial detection
mtcnn = MTCNN()

# Load ResNet model for facial recognition
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

# Obtain REFERENCE photo from camera
cam = cv2.VideoCapture(0)
input("Click enter when you wish to take the REFERENCE picture.\n")
_, opencv_image = cam.read()
print("DONE")
ref_embeddings = get_embeddings(opencv_image)

input("Click enter when you wish to take the NEW picture.\n")
_, opencv_image = cam.read()
print("DONE")
new_embeddings = get_embeddings(opencv_image)

cam.release()

# Print out feature vector (embeddings)
print("Reference embeddings:", ref_embeddings)
print("New embeddings:", new_embeddings)

# Compare face embeddings of original and new face
distance = (new_embeddings - ref_embeddings).norm().item()
if distance < 1.0:
    print("Verdict: Same person")
else:
    print("Verdict: Different person")
