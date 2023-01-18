import requests, json, threading, time, os, colab
from IPython.display import display
from IPython.display import HTML
from IPython.display import clear_output;
from compute import job_manager, job
import importlib
# import get_ipython()
API = "https://voidops.com/compute/api"
from IPython import get_ipython
def set_connection_status(uuid, msg, color, end = ""):
    display(HTML("%s <code><font color='%s'>%s</font></code>%s<br><hr><br>" % (msg, color, uuid, end)),  display_id = "void-connection")
def send(request_function, data):
    response = requests.post(f"{API}/{request_function}", json = data, headers={"User-Agent": "VOID-Compute-Client"})
    if response.status_code == 200:
        decoded = None
        if response.text == "":
            raise Exception("RecFFSDFeived empty response")
        else:
            try:
                decoded = json.loads(response.text)
            except:
                raise Exception("Couldn't parse response" + response.text)
        return decoded
    else:
        print("Couldn't post job submission: " + str(response))
    return None
def run(uuid):
    clear_output()
    set_connection_status(uuid, "Connecting to", "orange", "...")
    job_manager.run()
    while True:
        response = send("get_jobs", {"uuid": uuid})
        if response != None:
            if response["status"] == "ok":
                set_connection_status(uuid, "Currently working for ", "green")
                server_jobs = response["jobs"]
                num_jobs = len(server_jobs)
                if num_jobs > 0:
                    for _job in job_manager.running_jobs:
                        if not any(serverJobData["id"] == _job.data['id'] for serverJobData in server_jobs): # if job is not in server jobs
                            print("Signaling termination of %s" % _job.data['id'])
                            job_manager.signal_termination(_job.data['id'])
                    for jobData in response["jobs"]:
                        if jobData['status'] == "pending":
                            print("Adding to queue %s (Currently running: %s)" % (jobData['id'], len(job_manager.running_jobs)))
                            job_manager.add_to_queue(job.job(uuid, jobData)) 
            else:
                if response["code"] != 404:
                    display(HTML("<font color='red'>" + response["message"] + "</font>"), display_id = "void-error")
        else:
            set_connection_status(uuid, "Waiting for", "orange", "...")
        time.sleep(1)