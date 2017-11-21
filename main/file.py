import os
import shutil
import logging


class File:

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self._class_name = type(self).__name__

    @property
    def class_name(self):
        return self._class_name

    def remove(self):

        if self.path:
            if os.path.exists(self.path):

                try:
                    logging.warning(
                        "Removing resource: {} [{}].". format(self._class_name, self.path))
                    shutil.rmtree(self.path)

                except OSError:
                    logging.error(
                        "Could not remove resource: {} [{}].". format(self._class_name, self.path))
