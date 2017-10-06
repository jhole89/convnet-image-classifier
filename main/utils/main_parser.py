from argparse import ArgumentParser
import os


class MainParser(ArgumentParser):

    def __init__(self):
        super().__init__()

    def register_args(self):
        self.add_argument("img_dir", type=self.valid_path, help="Directory of images")
        self.add_argument("-c", "--clean", help="Retrain model from scratch", action='store_true')
        self.add_argument("-m", "--model_dir", default=os.path.abspath('tmp'),
                          help="Directory to store model and logs")
        self.add_argument("-i", "--iterations", type=self.positive_int,
                          default=2000, help="Number of training iterations")
        self.add_argument("-v", "--verbosity", type=str,
                          default="info", help="Verbosity level")
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
