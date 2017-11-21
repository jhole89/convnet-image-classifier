import os
import logging
from main.file import File


class Directory(File):

    def __init__(self, path):
        super().__init__(path)

    def create(self):

        if self.path:

            if os.path.exists(self.path):
                logging.debug(
                    "Directory [%s] already exists. Skipping create.", self.path)

            else:
                try:
                    logging.debug("Generating directory [%s].", self.path)
                    os.mkdir(self.path)

                except OSError:
                    logging.error(
                        "Could not generate directory [%s].", self.path)
