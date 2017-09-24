import argparse


class MainParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def register_args(self):
        self.parser.add_argument("d", "--img_dir", help="Directory of images")
        self.parser.add_argument("-c", "--clean", help="Retrain model from scratch", action='store_true')
        self.parser.add_argument("-m", "--model_dir", help="Directory to store model and logs")
        self.parser.add_argument("-i", "--iterations", help="Number of training iterations")
