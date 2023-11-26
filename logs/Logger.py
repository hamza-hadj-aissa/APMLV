import logging


class Logger:
    def __init__(self, name=__name__, level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create a console handler and set the level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Create a formatter and set the format
        formatter = logging.Formatter(
            '[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler("./logs/lvm_balancer.log")
        file_handler.setLevel(level)

        # Create a formatter and set the format for both handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add both handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger
