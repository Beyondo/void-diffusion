import sys
import socketserver
import time
from IPython import get_ipython
try:
  port = int(sys.argv[1])
  get_ipython().system("kill -9 $(lsof -t -i:%d) &> /dev/null" % port)
  class TCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip()
        print("{} wrote:".format(self.client_address[0]))
        print(self.data)
        # just send back the same data, but upper-cased
        self.request.sendall(self.data.upper())
  if __name__ == "__main__":
      HOST, PORT = "localhost", port
      with socketserver.TCPServer((HOST, PORT), TCPHandler) as server:
          # Activate the server; this will keep running until you
          # interrupt the program with Ctrl-C
          server.serve_forever()
except: print("Error occured.")