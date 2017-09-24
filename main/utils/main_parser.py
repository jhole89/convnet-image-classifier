from argparse import ArgumentParser


class MainParser(ArgumentParser):

    def __init__(self):
        super().__init__()

    def register_args(self):
        self.add_argument("-d", "--img_dir", help="Directory of images")
        self.add_argument("-c", "--clean", help="Retrain model from scratch", action='store_true')
        self.add_argument("-m", "--model_dir", help="Directory to store model and logs")
        self.add_argument("-i", "--iterations", type=int, help="Number of training iterations")
