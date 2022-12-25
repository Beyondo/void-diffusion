import requests, json, threading, time, os, colab
from IPython.display import display
from IPython.display import HTML
from IPython.display import clear_output;
from compute import job_manager, job
import importlib
# import get_ipython()
API = "https://voidops.com/diffusion/api.php"
from IPython import get_ipython
def set_connection_status(uuid, msg, color, end = ""):
    display(HTML("%s <code><font color='%s'>%s</font></code>%s<br><hr><br>" % (msg, color, uuid, end)),  display_id = "void-connection")
def send(data):
    response = requests.post(API, json = data, headers={"User-Agent": "VOID-Compute-Client"})
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        print("Couldn't post job submission: " + str(response))
    return None


def run(uuid):
    clear_output()
    set_connection_status(uuid, "Connecting to", "orange", "...")
    job_manager.run()
    while True:
        response = send({"uuid": uuid, "type": "get_jobs"})
        if response != None:
            if response["status"] == "ok":
                set_connection_status(uuid, "Currently working for ", "green")
                server_jobs = response["jobs"]
                num_jobs = len(server_jobs)
                if num_jobs > 0:
                    for _job in job_manager.currently_running:
                        if not any(serverJobData["id"] == _job.data['id'] for serverJobData in server_jobs): # if job is not in server jobs
                            print("Signaling termination...")
                            job_manager.signal_termination(uuid, _job.data['id'])
                    for jobData in response["jobs"]:
                        if jobData['status'] == "pending":
                            print("Adding to queue...")
                            job_manager.add_to_queue(job.job(uuid, jobData)) 
            else:
                if r["code"] != 404:
                    display(HTML("<font color='red'>" + response["message"] + "</font>"), display_id = "void-error")
        else:
            set_connection_status(uuid, "Waiting for", "orange", "...")
        time.sleep(1)