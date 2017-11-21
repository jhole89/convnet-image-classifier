import logging
import tensorflow as tf


class Logger:

    def __init__(self, log_level='INFO'):
        self.log_level = log_level.upper()
        self.format = '%(asctime)s %(levelname)s %(message)s'

    def set_log_level(self):
        if self.log_level in ['FATAL', 'ERROR', 'WARN', 'INFO', 'DEBUG']:

            log_degree = logging.getLevelName(self.log_level)

            logging.basicConfig(level=log_degree, format=self.format)
            tf.logging.set_verbosity(log_degree)

        else:
            raise ValueError
