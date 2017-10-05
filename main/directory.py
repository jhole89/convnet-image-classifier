import os
import re
import shutil
import logging
from random import random


class Directory:

    def __init__(self, directory='tmp'):
        self.directory = directory

    def remove(self):

        if self.directory:
            if os.path.exists(self.directory):

                try:
                    logging.warning(
                        "Removing resource: Directory [%s].", os.path.abspath(self.directory))
                    shutil.rmtree(self.directory)

                except OSError:
                    logging.error(
                        "Could not remove resource: Directory [%s].", os.path.abspath(self.directory))

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
