"""Camera demo"""
from PIL import Image

import cv2
import dlib
import numpy as np
import torch
from torchvision import transforms

from model import CFPD
from utils import transform_affine, transform_landmarks

MODEL_PATH = "./data/models/trained_using_augmented_images.pth"


def main():
    img_shape = np.asarray((112, 112))
    detector = dlib.get_frontal_face_detector()

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH)

    model = CFPD(False)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    show_bbox = False
    while True:
        _, img = cap.read()
        gray_scaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = detector(gray_scaled, 1)
        for detection in detections:
            resized_img, scale, offset, min_xy = prepare_image(gray_scaled, img_shape, data_transforms, detection,
                                                               device)
            center = img_shape / 2
            with torch.no_grad():
                output = model(resized_img)
                output = output.cpu().numpy().reshape(-1, 2)
            output = transform_landmarks(output, 0, 1/scale, offset+min_xy, center)

            for loc in range(0, len(output), 2):
                top_left = tuple(output[loc].astype(int))
                bottom_right = tuple(output[loc+1].astype(int))
                cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 1)
            if show_bbox:
                draw_detected_frame(img, detection, (0, 255, 0), extend=10)

        cv2.imshow("CFPD", img)
        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            break
        if key == ord("b"):
            show_bbox = not show_bbox
    cap.release()
    cv2.destroyAllWindows()


def prepare_image(img, img_shape, transformations, detection, device):
    """Prepare image"""
    resized_img, scale, offset, min_xy = preprocess(img, img_shape, detection)
    resized_img = Image.fromarray(resized_img)
    resized_img = transformations(resized_img)
    resized_img = resized_img.unsqueeze(0)
    return resized_img.to(device), scale, offset, min_xy


def preprocess(img, img_shape, detection, margin=0.25):
    """Preprocessing"""
    left = detection.left()
    right = detection.right()
    top = detection.top()
    bottom = detection.bottom()
    w, h = abs(left - right), abs(top - bottom)
    face_size = max(w, h)
    desired_size = 1 - 2 * margin
    desired_size *= min(img_shape)
    scale = desired_size / face_size

    img_center = np.asarray([x/2 for x in img.shape[:2]][::-1])
    center = np.asarray(((left+right)/2, (top+bottom)/2))
    offset = center - img_center

    resized_img = transform_affine(img, 0, scale, -offset, center)

    min_xy = (img_center - img_shape / 2).astype(int)
    max_xy = (img_center + img_shape / 2).astype(int)

    resized_img = resized_img[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0]]

    return resized_img, scale, offset, min_xy


def draw_detected_frame(img, detection, color, extend):
    """Draw detected face on image"""
    detection_left = detection.left() - extend
    detection_top = detection.top() - extend
    detection_right = detection.right() + extend
    detection_bottom = detection.bottom() + extend
    cv2.rectangle(img, (detection_left, detection_top), (detection_right, detection_bottom), color, 1)


if __name__ == "__main__":
    main()
