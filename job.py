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
        return client.send("update_job", data={"uuid": self.uuid, "job": self.data })

    def callback(self, progress, status):
        self.data['progress'] = progress
        self.data['status'] = status
        self.update()

    def process(self):
        if self.update():
            job_manager.running_jobs.append(self)
            try:
                #scriptFile = os.path.join("." + os.getcwd(), "scripts", self.data['script']) + ".py"
                #print(f"Processing in {scriptFile} ({self.data['id']})")
                importlib.import_module(".txt2img.py", package=".scripts")
                print("Imported")
            except Exception as e:
                self.data['status'] = "error"
                self.data['progress'] = -1
                print(e)
                job_manager.running_jobs.remove(self)
                return False
            #if self.data['script'] == "run_script":
            #    try:
            #        importlib.import_module(os.path.join("scripts", self.data['script']))
            #    except Exception as e:
            #        self.data['status'] = "error"
            #        self.data['progress'] = -1
            #        print(e)
            #        job_manager.running_jobs.remove(self)
            #        return False
            #elif self.data['type'] == "install_vendor":
            #    try:
            #        import env
            #        env.install_vendor()
            #    except Exception as e:
            #        self.data['status'] = "error"
            #        self.data['progress'] = -1
            #        print(e)
            #        job_manager.running_jobs.remove(self)
            #        return False
            #else:
            #    self.data['status'] = "error"
            #    self.data['progress'] = -1
            #    self.update()
            #    print("Unknown job type")
            #    job_manager.running_jobs.remove(self)
            #    return False
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