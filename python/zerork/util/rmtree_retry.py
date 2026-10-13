
import time
import shutil

def rmtree_retry(directory, ntries=8):
    for _ in range(ntries):
        try:
            return shutil.rmtree(directory)
        except OSError as e:
            if (e.strerror == "Directory not empty" or
               e.strerror == "Device or resource busy"):
                # wait a bit and try again
                time.sleep(0.05)
            else:
                raise
    raise RuntimeError(f"shutil.rmtree('{directory}') failed with ENOTEMPTY {ntries} times")

