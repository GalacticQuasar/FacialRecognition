from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
import cv2
import time

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Initialize OpenCV VideoCapture
cam = cv2.VideoCapture(0)

# Take picture
input("Click enter when you want to take the picture.")
result, opencv_image = cam.read()

# img = Image.open("<filename>.jpg")

# Convert opencv image to PIL image
color_converted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
img = Image.fromarray(color_converted)

# Clean up OpenCV resources
cam.release()

# Center crop image to square
if img.width > img.height:
    left = (img.width - img.height) / 2
    right = img.width - left
    img = img.crop(box=(left, 0, right, img.height))
else:
    top = (img.height - img.width) / 2
    bottom = img.height - top
    img = img.crop(box=(0, top, img.width, bottom))

# Downscale
multiplier = img.width / 160  # height and width are equal
downscaled_img = img.resize(size=(160, 160))  # Downscale Image
# Detect faces in the image
start_time = time.perf_counter()
boxes, _ = mtcnn.detect(downscaled_img)
print(time.perf_counter() - start_time)
# Resize boxes to map on original image size
if boxes is not None:
    boxes = [list(map(lambda x: x * multiplier, box)) for box in boxes]  # Upscale box coordinates for original image size
    for box in boxes:
        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(img)
        draw.rectangle(box, outline='red', width=3)

# Display image with detected faces
img.show()
