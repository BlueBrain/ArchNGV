import re
import sys
from contextlib import contextmanager


@contextmanager
def show_wait_message(mesg):
    print(mesg + " Please wait...", end="\r")
    sys.stdout.flush()
    yield
    print(" " * (len(mesg) + 15), end="\r")  # Clear


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    r = re.compile("(\d+)")
    return [atoi(c) for c in r.split(text)]
