from facenet_pytorch import MTCNN
from PIL import Image
import cv2
import time


def center_crop(img, dim):  # METHOD WRITTEN BY NANDAN MANJUNATHA
    # Returns cv2 image
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return crop_img


# Initialize MTCNN for face detection
mtcnn = MTCNN()
# mtcnn.cuda(device=0)  # TODO: Set device to use GPU
# print(mtcnn.device)

# Initialize OpenCV VideoCapture
cam = cv2.VideoCapture(0)  # Replace 0 with video file to run on video file

# Set image processing resolution
maxRes = min(cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
resolution = int(input(f"Input processing resolution (cannot be over {maxRes}): "))

while resolution > maxRes:
    resolution = int(input(f"ERROR. Input processing resolution (cannot be over {maxRes}): "))

while True:
    result, frame = cam.read()

    frame = center_crop(frame, (min(frame.shape[0], frame.shape[1]), min(frame.shape[0], frame.shape[1])))

    # Convert OpenCV Image to PIL
    color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(color_converted)

    # Downscale
    multiplier = pil_img.width / resolution  # height and width are equal
    downscaled_img = pil_img.resize(size=(resolution, resolution))

    # Detect faces in the image
    start_time = time.perf_counter()
    boxes, _ = mtcnn.detect(downscaled_img)
    print(1.0 / (time.perf_counter() - start_time))

    # Resize boxes to map on original image size
    if boxes is not None:
        boxes = [list(map(lambda x: x * multiplier, box)) for box in boxes]

        for box in boxes:
            # Draw bounding boxes on the image
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (128, 128, 0), 2)

    cv2.imshow('weedio', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up OpenCV Resources
cam.release()
cv2.destroyAllWindows()
