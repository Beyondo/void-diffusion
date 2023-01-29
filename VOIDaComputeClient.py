import requests, json, time, threading
import VOIDComputeShell, VOIDLogger
class VOIDComputeClient:
    API = "https://voidops.com/compute/api"
    def __init__(self, uuid):
        self.uuid = uuid
        self.shells = []
    def addShells(self, count):
        for i in range(count):
            self.shells.append(VOIDComputeShell.VOIDComputeShell(self))
    def send(self, request_function, data, retries = 0):
        functionUrl = f"{self.API}/{request_function}"
        response = requests.post(functionUrl, json = data, headers={"User-Agent": "VOID-Compute-Client"})
        decoded = None
        if response.status_code == 200:
            if response.text == "":
                VOIDLogger.Log(f"Client {self.uuid} ({request_function})", "Returned an empty HTTP response")
            else:
                try:
                    decoded = json.loads(response.text)
                    if decoded["status"] == "error":
                        VOIDLogger.Log(f"Client {self.uuid} ({request_function})", decoded["message"])
                except:
                    VOIDLogger.Log(f"Client {self.uuid} ({request_function})", "Returned an invalid HTTP response")
        elif response.status_code == 502:
            if retries > 5:
                VOIDLogger.Log(f"Client {self.uuid} ({request_function})", "Server is not responding")
            else:
                time.sleep(1)
                return self.send(request_function, data, retries + 1)
        else:
            VOIDLogger.Log(f"Client {self.uuid} ({request_function})", f"Returned an HTTP error ({response.status_code})")
        return decoded['output']

    def pinging_proces(self):
        while True:
            try:
                self.send("ping")
                time.sleep(3)
            except:
                VOIDLogger.Log(f"Client {self.uuid}", "Ping failed. Retrying in 3 seconds...")
                time.sleep(3)
    def start_pinging_async(self):
        t = threading.Thread(target=self.pinging_proces)
        t.start()