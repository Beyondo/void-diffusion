# A shell represents a compute thread that can be started, stopped, and sent data to.
# It interacts with the API directly in an infinite loop.
import importlib
class VOIDComputeShell():
    def __init__(self, client):
        super().__init__()
        self.client = client
        response = self.client.send("shell/register")
        if response["status"] == "error":
            raise Exception(f"Error registering shell ({self.id}): {response['message']}")
        self.id = response["shell_id"]
    def out(self, data):
        response = self.client.send(f"shell/{self.id}/out", data)
        if response['status'] == "error":
            raise Exception(f"Error sending to shell ({self.id}): {response['message']}")
    def run(self):
        try:
            self.status = "running"
            self.progress = 0
            self.out()
            print(f"Processing in {self.script} ({self.job_id})")
            # if self.script contains a dot, then it's a module in a package
            mod = None
            if "." in self.script:
                package, script = self.script.split(".")
                mod = importlib.import_module(f".{script}", package=package)
            else:
                mod = importlib.import_module(self.script)
            importlib.reload(mod)
            if mod.run(args=self.args, callback=self.callback):
                self.status = "complete"
                self.progress = 100
                self.update()
            else:
                self.status = "error"
                self.progress = -1
                self.update()
        except Exception as e:
            print("Exception: ", end="")
            print(e)
            self.status = "error"
            self.progress = -1
            self.update()
    def callback(self, progress, status):
        self.progress = progress
        self.status = status
        self.update()
    def stop(self):
        self.status = "stopped"
        self.update()