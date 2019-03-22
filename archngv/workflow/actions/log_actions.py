import time
import logging
import functools

L = logging.getLogger(__name__)


def log_start_end(func):
    @functools.wraps(func)
    def decorated_func(*args, **kwargs):

        L.info("Enter {}".format(func.__name__))

        result = func(*args, **kwargs)

        L.info("Exit {}".format(func.__name__))
        return result
    return decorated_func


def log_elapsed_time(func):
    @functools.wraps(func)
    def decorated_func(*args, **kwargs):

        t1 = time.time()

        result = func(*args, **kwargs)

        t2 = time.time()

        L.info("Elapsed time for {}: {}".format(func.__name__, t2 - t1))

        return result
    return decorated_func
