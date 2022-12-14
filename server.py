import sys
import socketserver
import time
from IPython import get_ipython
try:
  port = int(sys.argv[1])
  get_ipython().system('python3 -m http.server {} &'.format(port))
except Exception as e:
  print("Server error: ", e)
  sys.exit(1)