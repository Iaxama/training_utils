from logging.handlers import QueueHandler
import logging

def multiprocess_log_init(q):
    qh = QueueHandler(q)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(qh)
