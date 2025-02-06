import cv2
import numpy as np
from skimage.morphology import skeletonize


def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    r, g, b = cv2.split(img)

    output1_r = clahe.apply(r)
    output1_g = clahe.apply(g)
    output1_b = clahe.apply(b)
    frame_orig_clahe = cv2.merge((output1_r, output1_g, output1_b))

    return frame_orig_clahe


def draw_lanes(img, prediction, width, height):
    prediction = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_LANCZOS4)
    _, binary_mask = cv2.threshold(prediction, 0.5, 1.0, cv2.THRESH_BINARY)
    skeleton = skeletonize(binary_mask)
    skeleton = (skeleton * 255).astype(np.uint8)
    roi = skeleton[350:, :]
    skeleton_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in skeleton_contours:
        contour[:, :, 1] += 350
        if (cv2.arcLength(contour, True)) > 400:
            cv2.polylines(img, [contour], isClosed=False, color=(0, 255, 0), thickness=2)
    return img


def draw_lanes_basic(img, prediction, width, height):
    prediction = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_LANCZOS4)
    mask = (prediction > 0.5).astype(np.uint8)
    mask_resize = cv2.resize(mask, ((img.shape[1]), (img.shape[0])), interpolation=cv2.INTER_CUBIC)
    img[mask_resize == 1, :] = (255, 0, 0)
    return img


def draw_ego_lane(img, prediction, width, height):
    prediction = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_LANCZOS4)
    _, binary_mask = cv2.threshold(prediction, 0.5, 1.0, cv2.THRESH_BINARY)
    skeleton = skeletonize(binary_mask)
    skeleton = (skeleton * 255).astype(np.uint8)
    roi = skeleton[350:, :]
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        image_center_x = prediction.shape[1] // 2

        # Initialize variables
        left_ego_lane = None
        right_ego_lane = None
        min_left_distance = float('inf')
        min_right_distance = float('inf')

        for contour in contours:
            if (cv2.arcLength(contour, True)) > 400:
                x, y, w, h = cv2.boundingRect(contour)
                lane_center_x = x + w // 2

                # Classify contour as either left or right of the center
                if lane_center_x < image_center_x:
                    # Check if it's the closest to the center on the left side
                    distance_to_center = abs(image_center_x - lane_center_x)
                    if distance_to_center < min_left_distance:
                        min_left_distance = distance_to_center
                        left_ego_lane = contour
                elif lane_center_x > image_center_x:
                    # Check if it's the closest to the center on the right side
                    distance_to_center = abs(lane_center_x - image_center_x)
                    if distance_to_center < min_right_distance:
                        min_right_distance = distance_to_center
                        right_ego_lane = contour

        if left_ego_lane is not None:
            left_ego_lane[:, :, 1] += 350
            cv2.drawContours(img, [left_ego_lane], -1, (255, 0, 0), 4)

        if right_ego_lane is not None:
            right_ego_lane[:, :, 1] += 350
            cv2.drawContours(img, [right_ego_lane], -1, (0, 0, 255), 4)

    return img


def draw_vehicles(img, results, class_list, target_classes):
    filtered_detections = []
    cx, cy, x1, bly, x2, y2 = None, None, None, None, None, None
    if len(results.boxes) > 0:
        for detection in results.boxes:
            class_id = int(detection.cls)
            if class_id in target_classes:
                filtered_detections.append(detection)

        for det in filtered_detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            _, _, w, h = map(int, det.xywh[0])
            conf = det.conf[0]
            cx = x1 + w // 2
            cy = y1 + h // 2
            bly = y1+h          # box bottom left y coord
            class_id = class_list[int(det.cls)]
            track_id = int(det.id) if det.id is not None else ""

            # Draw bounding box and track ID
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            # cv2.circle(img, (cx, cy), 6, (0, 255, 255), -1)
            # cv2.circle(img, (x1, y1), 6, (255, 0, 0), -1)
            # cv2.circle(img, (x1, bly), 6, (0, 255, 0), -1)    # box bottom left coord
            # cv2.circle(img, (x2, y2), 6, (0, 0, 255), -1)      # box bottom right coord
            cv2.putText(img, f'ID: {track_id}, {class_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 1)

        return cx, cy, x1, bly, x2, y2, img

    else:
        return cx, cy, x1, bly, x2, y2, img


def ft(prediction, width, height):
    prediction = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_LANCZOS4)

    dft = np.fft.fft2(prediction, axes=(0, 1))
    dft_shift = np.fft.fftshift(dft)
    mask = np.zeros_like(prediction)
    mask2 = cv2.GaussianBlur(mask, (31, 31), 0)
    mask2 = cv2.GaussianBlur(mask2, (31, 31), 0)
    mask2 = cv2.GaussianBlur(mask2, (31, 31), 0)
    dft_shift_masked2 = np.multiply(dft_shift, mask) / 255
    back_ishift_masked2 = np.fft.ifftshift(dft_shift_masked2)
    img_filtered2 = np.fft.ifft2(back_ishift_masked2, axes=(0, 1))
    img_filtered2_final = np.abs(img_filtered2).clip(0, 255).astype(np.uint8)

    return img_filtered2_final
