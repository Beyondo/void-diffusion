import os, sys, time, threading
from IPython.display import display
from IPython.display import HTML
from IPython.display import clear_output;
import VOIDComputeClient, VOIDLogger
clients = []
def StartColabUnit(clientsInfo):
    clear_output()
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
    display(HTML("<h1>Starting VOID Compute Unit...</h1>"))
    for clientInfo in clientsInfo:
        clients.append(VOIDComputeClient.VOIDComputeClient(clientInfo[0]))
        clients[-1].addShells(clientInfo[1])
        clients[-1].start_pinging_async()
    while True:
        try:
            time.sleep(1)
            if len(threading.enumerate()) == 1:
                break
        except KeyboardInterrupt:
            VOIDLogger.Log("Compute Unit", "Keyboard interrupt")
            break
        except Exception as e:
            VOIDLogger.Log("Compute Unit", e)
            break

def StartLocalUnit(clientsInfo):
    clear_output()
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
    print("Starting VOID Compute Unit...")
    for clientInfo in clientsInfo:
        clients.append(VOIDComputeClient.VOIDComputeClient(clientInfo[0]))
        clients[-1].addShells(clientInfo[1])
        clients[-1].start_pinging_async()
    while True:
        try:
            time.sleep(1)
            if len(threading.enumerate()) == 1:
                break
        except KeyboardInterrupt:
            VOIDLogger.Log("Compute Unit", "Keyboard interrupt")
            break
        except Exception as e:
            VOIDLogger.Log("Compute Unit", e)
            break