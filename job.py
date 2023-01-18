import json, os, importlib
import client, job_manager
class job:
    def __init__(self, uuid, jobData):
        self.uuid = uuid
        self.data = jobData

    def submit(self):
        self.data['progres'] = 100
        self.data['status'] = "complete"
        return client.send("submit_job", data={"uuid": self.uuid, "job": self.data })

    def update(self):
        print("Updating job %s" % self.data['id'])
        return client.send("update_job", data={"uuid": self.uuid, "job": self.data })

    def callback(self, progress, status):
        self.data['progress'] = progress
        self.data['status'] = status
        self.update()

    def process(self):
        if self.update():
            job_manager.running_jobs.append(self)
            try:
                print(f"Processing in {self.data['script']} ({self.data['id']})")
                mod = importlib.import_module(self.data['script'])
                #importlib.reload(mod)
                if mod.run(args=self.data['args'], callback=self.callback):
                    self.data['status'] = "complete"
                    self.data['progress'] = 100
                else:
                    self.data['status'] = "error"
                    self.data['progress'] = -1
            except Exception as e:
                print("Exception: ", end="")
                print(e)
                self.data['status'] = "error"
                self.data['progress'] = -1
                job_manager.running_jobs.remove(self)
                return False
        else:
            self.data['status'] = "error"
            self.data['progress'] = -1
        self.update()
        job_manager.running_jobs.remove(self)

    def stop():
        self.data['status'] = "stopped"
        self.update()