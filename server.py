import sys
import socketserver
import time
from IPython import get_ipython
from IPython.display import Javascript
import os
try:
  port = int(sys.argv[1])
  os.system("python -m http.server " + str(port))
except Exception as e:
  print("Server error: ", e)
  sys.exit(1)