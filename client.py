import requests, json, threading, time, os, colab, sys
from IPython.display import display
from IPython.display import HTML
from IPython.display import clear_output;
import job_manager, job
import importlib
importlib.reload(job_manager)
importlib.reload(job)
# import get_ipython()
API = "https://voidops.com/compute/api"
from IPython import get_ipython
def set_connection_status(uuid, msg, color, end = ""):
    display(HTML("%s <code><font color='%s'>%s</font></code>%s<br><hr><br>" % (msg, color, uuid, end)),  display_id = "void-connection")
def send(request_function, data, retries = 0):
    functionUrl = f"{API}/{request_function}"
    response = requests.post(functionUrl, json = data, headers={"User-Agent": "VOID-Compute-Client"})
    decoded = None
    if response.status_code == 200:
        if response.text == "":
            raise Exception("Received an empty response from " + request_function)
        else:
            try:
                decoded = json.loads(response.text)
            except:
                raise Exception("Couldn't parse response from " + request_function + ": " + response.text)
    elif response.status_code == 502:
        if retries > 5:
            raise Exception("Backend is down.")
        else:
            print("Packet loss. Retrying~")
            time.sleep(1)
            return send(request_function, data, retries + 1)
    else:
        raise Exception("Couldn't send request: " + str(response))
    return decoded
def run(uuid):
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
    clear_output()
    set_connection_status(uuid, "Connecting to", "orange", "...")
    job_manager.run()
    while True:
        try:
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
                                job_manager.try_add(job.job(uuid, jobData))
                else:
                    if response["code"] != 404:
                        display(HTML("<font color='red'>" + response["message"] + "</font>"), display_id = "void-error")
            else:
                set_connection_status(uuid, "Waiting for", "orange", "...")
            send("ping", {"uuid": uuid})
            time.sleep(1)
        except KeyboardInterrupt:
            set_connection_status(uuid, "Connection was interrupted from", "red")
            break
        except:
            set_connection_status(uuid, "Connecting to", "orange", "...")
            time.sleep(3)