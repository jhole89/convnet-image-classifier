from main.utils.logger import Logger


def test_logger():

    default_logger = Logger()
    default_logger.set_log_level()

    error_logger = Logger(log_level='error')
    error_logger.set_log_level()

    assert default_logger.log_level == 'INFO'
    assert error_logger.log_level == 'ERROR'
