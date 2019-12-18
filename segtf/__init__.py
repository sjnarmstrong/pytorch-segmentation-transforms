import logging


class App(object):
    logger = None

    def __init__(self):
        self.init_logger()

    def init_logger(self, logger_name='cogdata', debug=True):
        if self.logger is not None:
            return

        logger = logging.getLogger(logger_name)

        if debug and logger.level == logging.NOTSET:
            logger.setLevel(logging.DEBUG)

        self.logger = logger


app = App()
