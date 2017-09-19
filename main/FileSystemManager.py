import os
import re
import shutil
import logging
import tarfile
from random import random


class FileSystemManager:

    def __init__(self, source_dir=None, model_dir=None):
        self.source_dir = source_dir
        self.model_dir = model_dir
        self.archive_dir = None

    def clean_run(self):
        """Remove model and data dirs for a clean run"""

        for directory in [self.model_dir, self.source_dir]:
            if directory:
                if os.path.exists(directory):
                    try:
                        logging.info("Removing resource: Directory [%s].", os.path.abspath(directory))
                        shutil.rmtree(directory)
                    except OSError:
                        logging.error(
                            "Could not remove resource: Directory [%s].", os.path.abspath(directory))

    def data_science_fs(self, category0, category1):
        """Makes data science file system for ML modelling"""

        for new_dir in ['train', 'predict']:
            for new_category in [category0, category1]:

                abspath_dir = os.path.abspath(os.path.join(self.source_dir, new_dir, new_category))

                logging.info(
                    "Creating resource: Directory [%s]", abspath_dir)
                os.makedirs(abspath_dir)

    def organise_files(self, directory, category_rules):
        """Flattens directory tree to single level"""

        predict_ratio = 0.1

        for root, dirs, files in os.walk(directory):
            for file in files:

                if re.compile(list(category_rules.values())[0]).match(file):

                    if random() < predict_ratio:
                        train_test_dir = 'predict/'

                    else:
                        train_test_dir = 'train/'

                    try:
                        logging.debug(
                            "Moving %s from %s to %s", file, root,
                            os.path.join(self.source_dir, train_test_dir, list(category_rules.keys())[0]))

                        os.rename(
                            os.path.join(root, file),
                            os.path.join(self.source_dir, train_test_dir, list(category_rules.keys())[0], file))

                    except OSError:
                        logging.error("Could not move %s ", os.path.join(root, file))

                elif re.compile(list(category_rules.values())[1]).match(file):

                    if random() < predict_ratio:
                        train_test_dir = 'predict/'

                    else:
                        train_test_dir = 'train/'

                    try:
                        logging.debug("Moving %s from %s to %s", file, root,
                                      os.path.join(self.source_dir, train_test_dir, list(category_rules.keys())[1]))

                        os.rename(
                            os.path.join(root, file),
                            os.path.join(self.source_dir, train_test_dir, list(category_rules.keys())[1], file))

                    except OSError:
                        logging.error("Could not move %s ", os.path.join(root, file))

                else:
                    logging.error("No files matching category regex")

        try:
            logging.info("Removing resource: Directory [%s].", os.path.abspath(self.archive_dir))
            shutil.rmtree(self.archive_dir)
        except OSError:
            logging.error("Could not remove resource: Directory [%s].", os.path.abspath(self.archive_dir))
