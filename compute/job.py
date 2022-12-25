import json, os, importlib
from compute import client, job_manager
class job:
    def __init__(self, uuid, jobData):
        self.uuid = uuid
        self.data = jobData

    def submit(self):
        self.data.progress = 100
        self.data.status = "complete"
        return client.send({"uuid": self.uuid, "job": self.data, "type": "submit_job"})

    def update(self):
        return client.send({"uuid": self.uuid, "job": self.data, "type": "update_job"})

    def callback(self, progress, status):
        self.data.progress = progress
        self.data.status = status
        self.update()

    def process(self):
        if self.update():
            job_manager.running_jobs.append(self)
            print("Processing job " + self.data["id"])
            if self.data['type'] == "run_script":
                try:
                    importlib.import_module(os.path.join("scripts", self.data['script']))
                except Exception as e:
                    self.data['status'] = "error"
                    self.data['progress'] = -1
                    print(e)
                    job_manager.running_jobs.remove(self)
                    return False
            elif self.data['type'] == "install_vendor":
                try:
                    import env
                    env.install_vendor()
                except Exception as e:
                    self.data['status'] = "error"
                    self.data['progress'] = -1
                    print(e)
                    job_manager.running_jobs.remove(self)
                    return False
            else:
                self.data['status'] = "error"
                self.data['progress'] = -1
                print("Unknown job type")
                job_manager.running_jobs.remove(self)
                return False
            self.data['status'] = "complete"
            self.data['progress'] = 100
        else:
            self.data['status'] = "error"
            self.data['progress'] = -1
        self.update()
        job_manager.running_jobs.remove(self)

    def stop():
        self.data['status'] = "stopped"
        self.update()