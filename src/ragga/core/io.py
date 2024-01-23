import sys
from io import StringIO


class store_stdout_stderr(object):  # noqa: N801, UP004
    sys = sys

    def __init__(self, outbuff: StringIO, errbuff: StringIO, disable: bool = False):  # noqa: FBT002, FBT001
        self.outbuff = outbuff
        self.errbuff = errbuff
        self.disable = disable

    # Oddly enough this works better than the contextlib version
    def __enter__(self):
        if self.disable:
            return self
        self.old_stdout = self.sys.stdout
        self.old_stderr = self.sys.stderr

        self.sys.stdout = self.outbuff
        self.sys.stderr = self.errbuff

        return self

    def __exit__(self, *_):
        if self.disable:
            return
        self.sys.stdout = self.old_stdout
        self.sys.stderr = self.old_stderr




