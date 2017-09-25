from argparse import ArgumentParser
import os


class MainParser(ArgumentParser):

    def __init__(self):
        super().__init__()

    def register_args(self):
        self.add_argument("img_dir", type=self.valid_path, help="Directory of images")
        self.add_argument("-c", "--clean", help="Retrain model from scratch", action='store_true')
        self.add_argument("-m", "--model_dir", default='tmp', help="Directory to store model and logs")
        self.add_argument("-i", "--iterations", type=self.positive_int,
                          default=2000, help="Number of training iterations")
        return self

    @staticmethod
    def valid_path(path):
        if not os.path.exists(path):
            raise OSError
        return path

    @staticmethod
    def positive_int(value):
        integer_val = int(value)
        if integer_val <= 0:
            raise ValueError
        return integer_val
