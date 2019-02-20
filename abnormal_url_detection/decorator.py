import logging
import time


def run_timer(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        func(*args, **kw)
        logging.info('Function [%s] run time is %.2f' % (func.__name__, time.time() - local_time))
    return wrapper

