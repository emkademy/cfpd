"""Constant Variables"""
import argparse
import ast
import configparser


class ConfigInitializer:
    selected_section = None

    def __init__(self, config_path):
        self.set_selected_section(config_path)

    def set_selected_section(self, config_path):
        if ConfigInitializer.selected_section is None:
            args = self.parse_args()
            section_name = self.get_section_name(args)
            ConfigInitializer.selected_section = self.get_selected_section_values(config_path, section_name)

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--section-name', type=str, default=None, dest='section_name',
                            help='Type the name of the config.ini file section. Leave blank for default values')
        args = parser.parse_args()
        return args

    @staticmethod
    def get_section_name(args):
        section_name = args.section_name
        if section_name is None:
            section_name = "default_section"
        return section_name

    @staticmethod
    def get_selected_section_values(config_path, section_name):
        config = configparser.ConfigParser()
        config.read(config_path)
        config_selected = config[section_name]
        return config_selected


class ConfigParser:
    def __init__(self, config_path):
        config = ConfigInitializer(config_path)
        self.selected_section = config.selected_section

        self.type_to_converter = {
            int: self.config_str_to_int_converter,
            float: self.config_str_to_float_converter,
            list: self.config_str_to_list_converter,
            bool: self.config_str_to_bool_converter,
            tuple: self.config_str_to_tuple_converter,
            str: self.config_str_to_str_converter
        }

    @staticmethod
    def config_str_to_int_converter(selected_section, variable_name):
        return selected_section.getint(variable_name)

    @staticmethod
    def config_str_to_float_converter(selected_section, variable_name):
        return selected_section.getfloat(variable_name)

    @staticmethod
    def config_str_to_list_converter(selected_section, variable_name):
        return selected_section.get(variable_name).split(", ")

    @staticmethod
    def config_str_to_tuple_converter(selected_section, variable_name):
        return ast.literal_eval(selected_section.get(variable_name))

    @staticmethod
    def config_str_to_bool_converter(selected_section, variable_name):
        return selected_section.getboolean(variable_name)

    @staticmethod
    def config_str_to_str_converter(selected_section, variable_name):
        return selected_section.get(variable_name)


class Parameters:
    def __init__(self, config_path):
        self._config = ConfigParser(config_path)
        self._parameter_names_to_types = self.connect_parameter_names_to_types()

    @staticmethod
    def connect_parameter_names_to_types():
        parameter_names_to_types = {
            "batch_size": int,
            "shuffle": bool,
            "num_workers": int,
            "gpu": bool,
            "num_epochs": int,
            "learning_rate": float,
            "save_model_to": str,
            "start_epoch": int,
            "n_io_points": int,
            "n_units": int,
            "len_f_vector": int,
            "scheduler_step_size": int,
            "scheduler_gamma": float,

            "img_shape": tuple,
            "color": bool,
            "data_preprocessing_output_dir": str,

            "mirror": bool,
            "n_augmented_images": int,
            "train_dirs": list,
            "train_csv": str,
            "transformation_params": tuple,
            "test_fraction": float,
            "parts_order": str,

            "commonset_dirs": list,
            "commonset_csv": str,

            "challengingset_dirs": list,
            "challengingset_csv": str,

            "w300set_dirs": list,
            "w300set_csv": str,
        }
        return parameter_names_to_types

    def __getattr__(self, parameter_name):
        if parameter_name not in self._parameter_names_to_types:
            raise ValueError(f"{parameter_name} is not a recognized parameter")
        parameter_type = self._parameter_names_to_types[parameter_name]
        converter = self._config.type_to_converter[parameter_type]
        parameter_value = converter(self._config.selected_section, parameter_name)
        setattr(self, parameter_name, parameter_value)
        return parameter_value
