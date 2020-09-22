import os
import sys

additional_path = os.path.realpath('../')
if additional_path not in sys.path:
    sys.path.append(additional_path)
