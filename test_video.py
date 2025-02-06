import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import utils
from skimage.morphology import skeletonize

WIDTH = 1280
HEIGHT = 720
INPUT_WIDTH = 256
INPUT_HEIGHT = 256
TARGET_CLASSES = [2, 7]

coco_class_list = open("coco.txt", "r")
data = coco_class_list.read()
class_list = data.split("\n")

# model = load_model('model/model_for_lane_tuned_adam_binary_batch16_v2.keras')
# model = load_model('model/model_for_lane_tuned_adam_binary_batch16_resnet50_unet_fixed.keras')
# model = load_model('model/model_unet_aug_att.keras')
model = load_model('model/model_unet_att.keras')
# model = load_model('model/model_unet_resnet50_unet_att_conv2d.keras')
# model = load_model('model/model_unet_resnet50_unet_aug_att_conv2d.keras')
model_YOLO = YOLO("model/yolov10n.pt")

cap = cv2.VideoCapture("D:/Programozás/Python/Lane_detection/videos/tusimple_video.avi")

while cap.isOpened():
    success, frame_orig = cap.read()
    if success:
        frame_orig = cv2.resize(frame_orig, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        print(f"Frame original shape: {frame_orig.shape}")
        frame_orig_blurred = cv2.GaussianBlur(frame_orig, (5, 5), 0)

        frame = cv2.resize(frame_orig_blurred, (INPUT_WIDTH, INPUT_HEIGHT))
        print(f"Frame resized shape: {frame.shape}")
        frame = np.expand_dims(frame, axis=0)
        frame = frame / 255.0
        print(f"Frame expanded shape: {frame.shape}")
        lane_results = model.predict(frame)
        print(f"Results original shape: {lane_results.shape}")
        lane_results = np.squeeze(lane_results, axis=0)
        print(f"Results squeezed shape: {lane_results.shape}")

        vehicle_results = model_YOLO.track(frame_orig, persist=True)

        frame_with_lanes = utils.draw_ego_lane(frame_orig, lane_results, WIDTH, HEIGHT)

        cx, cy, x1, bly, x2, y2, img = utils.draw_vehicles(frame_with_lanes, vehicle_results[0], class_list, TARGET_CLASSES)
        cv2.imshow("pred", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
