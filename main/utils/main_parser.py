from argparse import ArgumentParser
import os


class MainParser(ArgumentParser):

    def __init__(self):
        super().__init__()

    def register_args(self):

        self.add_argument("img_dir", type=self.valid_path, help="Path of images")

        self.add_argument("-m", "--mode", type=self.mode_type,
                          default="predict", help="train or predict mode")

        self.add_argument("-d", "--model_dir", default=os.path.abspath('tmp'),
                          help="Directory to store/access model and logs")

        self.add_argument("-i", "--iterations", type=self.positive_int,
                          default=50, help="Number of iterations used in train mode")

        self.add_argument("-b", "--batch_size", type=self.positive_int,
                          default=1, help="Number of images to process in a single batch")

        self.add_argument("-v", "--verbosity", type=self.log_levels, default="info", help="Verbosity level")

        self.add_argument("-c", "--clean", help="Retrain model from scratch", action='store_true')

        return self

    @staticmethod
    def valid_path(value):
        abs_path = os.path.abspath(value)
        if not os.path.exists(abs_path):
            raise OSError
        return abs_path

    @staticmethod
    def positive_int(value):
        integer_val = int(value)
        if integer_val <= 0:
            raise ValueError
        return integer_val

    @staticmethod
    def mode_type(value):
        value = str(value).lower()
        if value not in ['train', 'predict']:
            raise ValueError
        return value

    @staticmethod
    def log_levels(value):
        value = str(value).upper()
        if value not in ['FATAL', 'ERROR', 'WARN', 'INFO', 'DEBUG']:
            raise ValueError
        return value
