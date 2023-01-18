import json, os, importlib
import client, job_manager
class job:
    def __init__(self, uuid, jobData):
        self.uuid = uuid
        self.data = jobData
    def signal_start(self):
        job_manager.running_jobs.append(self)
        self.data['status'] = "running"
        self.data['progress'] = 0
        return client.send("update_job", data={"uuid": self.uuid, "job": self.data })
    def signal_completion(self):
        job_manager.running_jobs.remove(self)
        self.data['progres'] = 100
        self.data['status'] = "complete"
        return client.send("update_job", data={"uuid": self.uuid, "job": self.data })
    def signal_failure(self):
        job_manager.running_jobs.remove(self)
        print("Job failed")
        self.data['progres'] = -1
        self.data['status'] = "error"
        return client.send("update_job", data={"uuid": self.uuid, "job": self.data })
    def update(self):
        resp = client.send("update_job", data={"uuid": self.uuid, "job": self.data })
        if resp['status'] == "error":
            raise Exception("Error updating job: " + resp['message'])

    def callback(self, progress, status):
        self.data['progress'] = progress
        self.data['status'] = status
        self.update()

    def process(self):
        try:
            self.signal_start()
            print(f"Processing in {self.data['script']} ({self.data['id']})")
            mod = importlib.import_module(self.data['script'])
            importlib.reload(mod)
            if mod.run(args=self.data['args'], callback=self.callback):
                self.signal_completion()
            else:
                self.signal_failure()
        except Exception as e:
            print("Exception: ", end="")
            print(e)
            self.signal_failure()

    def stop():
        self.data['status'] = "stopped"
        self.update()