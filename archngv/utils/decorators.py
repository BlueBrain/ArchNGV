# SPDX-License-Identifier: Apache-2.0

""" Decorators """
import logging
import time
from functools import wraps

L = logging.getLogger(__name__)


def log_execution_time(func):
    """Logs execution time of a function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        L.info("%s ran in %d s.", func.__name__, round(end - start, 2))
        return result

    return wrapper


def log_start_end(func):
    """Logs start and end of a function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        L.info("%s started.", func.__name__)
        result = func(*args, **kwargs)
        L.info("%s finished.", func.__name__)
        return result

    return wrapper
