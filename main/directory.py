import os
import shutil
import logging


class Directory:

    def __init__(self, path='tmp'):
        self.path = path

    def remove(self):

        if self.path:
            if os.path.exists(self.path):

                try:
                    logging.warning(
                        "Removing resource: Directory [%s].", os.path.abspath(self.path))
                    shutil.rmtree(self.path)

                except OSError:
                    logging.error(
                        "Could not remove resource: Directory [%s].", os.path.abspath(self.path))

    def create(self):

        if self.path:

            if os.path.exists(self.path):
                logging.debug(
                    "Directory [%s] already exists. Skipping create.", os.path.abspath(self.path))

            else:
                try:
                    logging.debug("Generating directory [%s].", os.path.abspath(self.path))
                    os.mkdir(self.path)

                except OSError:
                    logging.error(
                        "Could not generate directory [%s].", os.path.abspath(self.path))
