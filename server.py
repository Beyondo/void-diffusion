import sys
import socketserver
import time
from IPython import get_ipython
try:
  port = int(sys.argv[1])
  get_ipython().system_raw('python3 -m http.server {} &'.format(port))