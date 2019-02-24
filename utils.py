"""
Utility functions
"""
import numbers
import os

import cv2
import numpy as np


def transform_landmarks(data, angle, scale, translation, center):
    """
    Landmark transform
    """
    translation, center = validate_transform_params(angle, scale, translation, center)
    data_copy = data.copy()
    angle = np.deg2rad(-angle)

    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])

    scale_matrix = np.array([
        [scale, 0],
        [0, scale]
    ])

    transformation_matrix = np.dot(scale_matrix, rotation_matrix)
    data_copy -= center
    data_copy = np.dot(data_copy, transformation_matrix)
    data_copy += center + np.asarray(translation)
    return data_copy


def transform_affine(data, angle, scale, translation, center):
    """
    2D affine transformation
    """
    translation, center = validate_transform_params(angle, scale, translation, center)
    angle = np.deg2rad(angle + 90)
    data_copy = data.copy()

    translation_matrix = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
    ])

    alpha = scale * np.sin(angle)
    beta = scale * np.cos(angle)

    # noinspection PyTypeChecker
    scale_rotation_matrix = np.array([
        [alpha, beta, (1-alpha)*center[0] - beta*center[1]],
        [-beta, alpha, beta*center[0] + (1-alpha)*center[1]],
        [0, 0, 1]
    ])

    transformation_matrix = np.dot(translation_matrix, scale_rotation_matrix)

    data_copy = cv2.warpAffine(data_copy, transformation_matrix, data_copy.shape[:2][::-1])
    return data_copy


def validate_transform_params(angle, scale, translation, center):
    assert isinstance(angle, numbers.Number), "angle parameter must be a number"
    assert isinstance(scale, numbers.Number), "scale parameter must be a number"

    if isinstance(center, numbers.Number):
        center = (center, center)
    assert len(center) == 2,\
        "center parameter must be a number or 2D array"

    if isinstance(translation, numbers.Number):
        translation = (translation, translation)
    assert len(translation) == 2,\
        "translation parameter must be a number or 2D array"

    return translation, center


def load_pts_file(path):
    """ Load .pts file"""
    landmarks = np.genfromtxt(path, skip_header=3, skip_footer=1)
    return landmarks


def save_landmarks_as_pts_file(landmarks, path):
    """Save landmark coordinates as .pts file"""
    landmarks_pts = "version: 1\nn_points: 68\n{\n"
    for pts in landmarks:
        landmarks_pts += str(pts[0]) + " " + str(pts[1]) + "\n"
    landmarks_pts += "}"
    with open(path, "w") as file_:
        print(landmarks_pts, file=file_)


def makedir(path):
    if not is_exists(path):
        os.makedirs(path)


def is_exists(path):
    return os.path.exists(path)


def save_image(img, save_path):
    dir_name = os.path.dirname(save_path)
    if not is_exists(dir_name):
        makedir(dir_name)
    cv2.imwrite(save_path, img)


def get_face_parts_to_indices():
    face_parts_to_indices = {
        "left_chin": np.arange(8),
        "right_chin": np.arange(9, 17),
        "chin": np.arange(17),
        "left_eyebrow": np.arange(17, 22),
        "right_eyebrow": np.arange(22, 27),
        "eyebrows": np.arange(17, 27),
        "left_nose": np.asarray([31, 32]),
        "right_nose": np.asarray([34, 35]),
        "nose": np.arange(27, 36),
        "left_eye": np.arange(36, 42),
        "right_eye": np.arange(42, 48),
        "eyes": np.arange(36, 48),
        "left_mouth": np.asarray([48, 49, 50, 58, 59, 60, 61, 67]),
        "right_mouth": np.asarray([54, 53, 52, 56, 55, 64, 63, 65]),
        "mouth": np.arange(48, 68)
    }
    return face_parts_to_indices


def mirror_landmarks(landmarks, img_shape):
    """Mirror landmarks carefully"""
    landmarks_copy = landmarks.copy()
    face_parts_to_indices = get_face_parts_to_indices()
    indices = face_parts_to_indices  # To make its name shorter

    landmarks_copy[:, 0] = img_shape[1] - landmarks_copy[:, 0]

    left_eye_to_right_eye = np.concatenate((indices["left_eye"], indices["right_eye"]))
    right_eye_to_left_eye = np.concatenate((indices["right_eye"], indices["left_eye"]))
    landmarks_copy[left_eye_to_right_eye] = landmarks_copy[right_eye_to_left_eye]

    landmarks_copy[[36, 37, 41, 39, 38, 40]] = landmarks_copy[[39, 38, 40, 36, 37, 41]]
    landmarks_copy[[42, 43, 47, 45, 44, 46]] = landmarks_copy[[45, 44, 46, 42, 43, 47]]

    left_eyebrow_to_right_eyebrow = np.concatenate((indices["left_eyebrow"], indices["right_eyebrow"]))
    right_eyebrow_to_left_eyebrow = np.concatenate((indices["right_eyebrow"], indices["left_eyebrow"]))
    landmarks_copy[left_eyebrow_to_right_eyebrow] = landmarks_copy[right_eyebrow_to_left_eyebrow]

    left_nose_to_right_nose = np.concatenate((indices["left_nose"], indices["right_nose"]))
    right_nose_to_left_nose = np.concatenate((indices["right_nose"][::-1], indices["left_nose"][::-1]))
    landmarks_copy[left_nose_to_right_nose] = landmarks_copy[right_nose_to_left_nose]

    left_mouth_to_right_mouth = np.concatenate((indices["left_mouth"], indices["right_mouth"]))
    right_mouth_to_left_mouth = np.concatenate((indices["right_mouth"], indices["left_mouth"]))
    landmarks_copy[left_mouth_to_right_mouth] = landmarks_copy[right_mouth_to_left_mouth]

    left_chin_to_right_chin = np.concatenate((indices["left_chin"], indices["right_chin"]))
    right_chin_to_left_chin = np.concatenate((indices["right_chin"][::-1], indices["left_chin"][::-1]))
    landmarks_copy[left_chin_to_right_chin] = landmarks_copy[right_chin_to_left_chin]

    return landmarks_copy
