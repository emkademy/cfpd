"""Data preprocessing"""
from collections import defaultdict, namedtuple
import os

import cv2
import numpy as np
import pandas as pd

import utils


class DataPreprocessing:
    """Preprocessing base class. Since resizing is necessary for both train and test set, it is defined here"""
    def __init__(self, dataset_parameters, base_csv, dataset_dirs):
        self.dataset_parameters = dataset_parameters
        self.dataset_parameters.img_shape = np.asarray(self.dataset_parameters.img_shape)
        self.dataset_base_dir = "./data/images/"
        self.dataset_dirs = dataset_dirs
        self.base_csv = base_csv
        self.transformation_parameters = namedtuple("Transformation", ["center", "angle", "scale", "offset"])
        utils.makedir(dataset_parameters.data_preprocessing_output_dir)

    def resize_and_center_data(self):
        if utils.is_exists(self.base_csv):
            resized_df = pd.read_csv(self.base_csv)
        else:
            print("Resizing and centering data...")
            resized_img_paths = []
            resized_landmarks_paths = []
            img_paths = self.get_image_paths(self.dataset_dirs)
            for img_path in img_paths:
                img, landmarks = self.get_image_and_landmarks(img_path)
                resizing_parameters = self.get_resizing_transformation_parameters(img, landmarks)
                resized_img, resized_landmarks = self.resize_img_and_landmarks(img, landmarks, resizing_parameters)

                img_save_path = img_path.replace(self.dataset_base_dir,
                                                 self.dataset_parameters.data_preprocessing_output_dir)
                utils.save_image(resized_img, img_save_path)

                landmarks_save_path = img_save_path[:-4]+".pts"
                utils.save_landmarks_as_pts_file(resized_landmarks, landmarks_save_path)

                resized_img_paths.append(img_save_path)
                resized_landmarks_paths.append(landmarks_save_path)

            data_dict = {"image": resized_img_paths, "landmarks": resized_landmarks_paths}
            resized_df = self.save_csv(data_dict, self.base_csv)
        return resized_df

    @staticmethod
    def get_image_paths(dataset_dirs):
        img_paths = []
        for dir_ in dataset_dirs:
            files = os.listdir(dir_)
            for file_ in files:
                if ".pts" not in file_:
                    full_path = os.path.join(dir_, file_)
                    img_paths.append(full_path)
        return img_paths

    @staticmethod
    def get_image_and_landmarks(img_path):
        img = cv2.imread(img_path)
        landmarks = utils.load_pts_file(img_path[:-4] + ".pts")
        return img, landmarks

    def get_resizing_transformation_parameters(self, img, landmarks):
        center = np.mean(landmarks, axis=0)
        img_center = np.asarray([x / 2 for x in img.shape[:2]][::-1])
        offset = img_center - center

        face_size = max(np.max(landmarks, axis=0) - np.min(landmarks, axis=0))
        margin = 0.25  # We want face to be centered
        desired_size = 1 - 2 * margin
        desired_size *= min(self.dataset_parameters.img_shape)
        scale = desired_size / face_size
        angle = 0

        params = self.transformation_parameters(center, angle, scale, offset)
        return params

    def resize_img_and_landmarks(self, img, landmarks, resizing_parameters):
        transformed_img, transformed_landmarks = self.transform_img_and_landmarks(img, landmarks, resizing_parameters)

        img_center = np.asarray([x / 2 for x in img.shape[:2]][::-1])
        target_img_shape = self.dataset_parameters.img_shape
        min_xy = (img_center - target_img_shape / 2).astype(int)
        max_xy = (img_center + target_img_shape / 2).astype(int)

        resized_img = transformed_img[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0]]
        transformed_landmarks -= min_xy

        return resized_img, transformed_landmarks

    @staticmethod
    def transform_img_and_landmarks(img, landmarks, transformation_parameters):
        center = transformation_parameters.center
        angle = transformation_parameters.angle
        scale = transformation_parameters.scale
        offset = transformation_parameters.offset

        transformed_landmarks = utils.transform_landmarks(landmarks, angle, scale, offset, center)
        transformed_img = utils.transform_affine(img, angle, scale, offset, center)
        return transformed_img, transformed_landmarks

    @staticmethod
    def save_csv(data_dict, path_to_save):
        df = pd.DataFrame(data_dict)
        df.to_csv(path_to_save, index=None, header=True)
        return df

    def process_images(self):
        raise NotImplementedError


class TestsetPreprocessing(DataPreprocessing):
    def __init__(self, dataset_parameters, base_csv, dataset_dirs):
        super(TestsetPreprocessing, self).__init__(dataset_parameters, base_csv, dataset_dirs)

    def process_images(self):
        _ = self.resize_and_center_data()
    

class TrainsetPreprocessing(DataPreprocessing):
    """
    2) Normalize scaled images to canonical pose (only for training to investigate its impact).
    3) Augment scaled images by randomly scaling, rotating, translating (only for training to investigate its impact).
    """
    def __init__(self, dataset_parameters, base_csv, dataset_dirs):
        super(TrainsetPreprocessing, self).__init__(dataset_parameters, base_csv, dataset_dirs)

    def process_images(self):
        resized_df = self.resize_and_center_data()
        resized_df = self.mirror_and_save_data(resized_df)
        try:
            normalized_df = self.normalize_to_canonical_shape(resized_df)
            self.augment_data(normalized_df, resized_df)
        except RuntimeError:
            pass  # We don't want neither normalization nor augmentation. But that's still ok.

    def mirror_and_save_data(self, resized_df):
        mirrored_img_paths = []
        mirrored_landmarks_paths = []
        if self.dataset_parameters.mirror:
            for img_path in resized_df["image"]:
                resized_img, resized_landmarks = self.get_image_and_landmarks(img_path)

                mirrored_img, mirrored_landmarks = self.mirror_data(resized_img, resized_landmarks)
                mirrored_img_path, mirrored_landmarks_path = self.save_data(mirrored_img, mirrored_landmarks, img_path)

                mirrored_img_paths.append(mirrored_img_path)
                mirrored_landmarks_paths.append(mirrored_landmarks_path)

        resized_mirrored_img_paths = list(resized_df["image"]) + mirrored_img_paths
        resized_mirrored_landmarks_paths = list(resized_df["landmarks"]) + mirrored_landmarks_paths
        data_dict = {"image": resized_mirrored_img_paths, "landmarks": resized_mirrored_landmarks_paths}
        resized_df = self.save_csv(data_dict, self.base_csv)
        return resized_df

    @staticmethod
    def mirror_data(img, landmarks):
        mirrored_img = np.fliplr(img.copy())
        mirrored_landmarks = utils.mirror_landmarks(landmarks, mirrored_img.shape)
        return mirrored_img, mirrored_landmarks

    @staticmethod
    def save_data(img, landmarks, path):
        img_save_path = path[:-4] + "m" + path[-4:]
        utils.save_image(img, img_save_path)

        landmarks_save_path = path[:-4] + "m.pts"
        utils.save_landmarks_as_pts_file(landmarks, landmarks_save_path)

        return img_save_path, landmarks_save_path

    def normalize_to_canonical_shape(self, resized_df):
        normalized_path = self.base_csv.replace(".csv", "_normalized.csv")
        if utils.is_exists(normalized_path):
            normalized_df = pd.read_csv(normalized_path)
        elif self.dataset_parameters.n_augmented_images > 0:
            print("Normalizing images to canonical pose...")
            resized_img_paths = resized_df["image"]
            data_dict = defaultdict(lambda: [])
            img_idx = 1
            for path in resized_img_paths:
                resized_img, resized_landmarks = self.get_image_and_landmarks(path)
                normalization_parameters = self.get_normalization_transformation_parameters(resized_landmarks)
                normalized_img, normalized_landmarks = self.transform_img_and_landmarks(resized_img, resized_landmarks,
                                                                                        normalization_parameters)

                img_extension = path[-4:]
                path_without_extension = path[:-4]
                save_path_template = path_without_extension + "__{}"
                save_data_to = save_path_template.format(str(img_idx) + img_extension)
                normalized_img_path, normalized_landmarks_path = self.save_data(normalized_img, normalized_landmarks,
                                                                                save_data_to)

                data_dict["image"].append(normalized_img_path)
                data_dict["landmarks"].append(normalized_landmarks_path)

            normalized_df = self.save_csv(data_dict, normalized_path)
        else:
            raise RuntimeError
        return normalized_df

    def get_normalization_transformation_parameters(self, landmarks):
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        d_y = right_eye_center[1] - left_eye_center[1]
        d_x = right_eye_center[0] - left_eye_center[0]
        angle = -np.degrees(np.arctan2(d_y, d_x))

        center = np.mean(landmarks, axis=0)
        offset = 0
        scale = 1

        params = self.transformation_parameters(center, angle, scale, offset)
        return params

    def augment_data(self, normalized_df, resized_df):
        """Augments images in the dataset by randomly scaling, rotating and translating. Random samples are
        taken from normal distribution"""
        augmented_path = self.base_csv.replace(".csv", "_augmented.csv")
        if utils.is_exists(augmented_path):
            return
        elif self.dataset_parameters.n_augmented_images > 1:
            notice = ("Data augmentation is being performed. This may take a while according to the number of images"
                      " and n_augmented_images parameter...")
            print(notice)
            data_dict = defaultdict(lambda: [])
            normalized_img_paths = normalized_df["image"]

            transformation_params = self.dataset_parameters.transformation_params
            translation_std = np.asarray(transformation_params[:2])*self.dataset_parameters.img_shape
            scale_std = transformation_params[2]
            rotation_std = transformation_params[3]

            for path in normalized_img_paths:
                normalized_img, normalized_landmarks = self.get_image_and_landmarks(path)
                img_idx = 2
                for _ in range(self.dataset_parameters.n_augmented_images-1):
                    augmentation_parameters = self.get_augmentation_transformation_parameters(rotation_std, scale_std,
                                                                                              translation_std)

                    augmented_img, augmented_landmarks = self.transform_img_and_landmarks(normalized_img,
                                                                                          normalized_landmarks,
                                                                                          augmentation_parameters)

                    img_extension = path[-4:]
                    save_path = path[:-5] + str(img_idx) + img_extension
                    augmented_img_path, augmented_landmarks_path = self.save_data(augmented_img, augmented_landmarks,
                                                                                  save_path)

                    data_dict["image"].append(augmented_img_path)
                    data_dict["landmarks"].append(augmented_landmarks_path)
                    img_idx += 1

            data_dict["image"].extend(list(normalized_df["image"]))
            data_dict["landmarks"].extend(list(normalized_df["landmarks"]))

            data_dict["image"].extend(list(resized_df["image"]))
            data_dict["landmarks"].extend(list(resized_df["landmarks"]))

            self.save_csv(data_dict, augmented_path)

    def get_augmentation_transformation_parameters(self, rotation_std, scale_std, translation_std):
        angle = np.random.normal(0, rotation_std)
        offset = (np.random.normal(0, translation_std[0]), np.random.normal(0, translation_std[1]))
        scale = np.random.normal(1, scale_std)
        center = tuple(self.dataset_parameters.img_shape / 2)

        params = self.transformation_parameters(center, angle, scale, offset)
        return params
