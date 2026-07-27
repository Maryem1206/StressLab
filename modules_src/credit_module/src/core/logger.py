
import logging
def get_logger(name="credit"):
    l = logging.getLogger(name)
    l.setLevel(logging.INFO)
    return l
