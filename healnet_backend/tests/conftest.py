import logging
import os

# Prevent Python 3.14 access violation during asyncio task resolution in logging
logging.logAsyncioTasks = False

def pytest_sessionfinish(session, exitstatus):
    """
    On Python 3.14 Windows, scikit-learn / OpenMP C extensions hit an access violation
    during standard CPython DLL unloading on interpreter shutdown. Using os._exit bypasses
    the faulty DLL finalizer while cleanly returning the correct pytest exit code.
    """
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exitstatus)
