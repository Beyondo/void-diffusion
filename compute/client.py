import requests, json, threading, time, colab
from IPython.display import display
from IPython.display import HTML
from IPython.display import clear_output;
# import get_ipython()
API = "https://voidops.com/diffusion/api.php"
from IPython import get_ipython
def process_job(job):
    # This is where you put your code to process the job
    pass
def set_connection_status(uuid, msg, color, end = ""):
    display(HTML("%s <code><font color='%s'>%s</font></code>%s<br><hr><br>" % (msg, color, uuid, end)),  display_id = "void-connection")
def run(uuid):
    clear_output()
    set_connection_status(uuid, "Connecting to", "orange", "...")
    while True:
        response = requests.post(API, json = {"type": "get_jobs", "uuid": uuid}, headers={"User-Agent": "VOID-Compute-Client"})
        if response.status_code == 200:
            if response.text != "":
              json = json.loads(response.text)
              if json["status"] == "ok":
                  set_connection_status(uuid, "Currently working for ", "green")
                  num_jobs = len(json["jobs"])
                  if num_jobs > 0:
                    print("Processing %d jobs..." % num_jobs)
                    for job in json["jobs"]:
                        if job['status'] == "pending":
                            # Send a post request to the server
                            job['status'] = "processing"
                            job['progress'] = 0
                            response = requests.post(API, json = {"uuid": uuid, "job": job, "type": "update_job"}, headers={"User-Agent": "VOID-Compute-Client"})
                            if response.status_code == 200:
                                json = json.loads(response.text)
                                if json["status"] == "ok":
                                    if(process_job(job)):
                                        job['status'] = "complete"
                                        job['progress'] = 100
                                        requests.post(API, json = {"uuid": uuid, "job": job, "type": "update_job"}, headers={"User-Agent": "VOID-Compute-Client"})
                                    else
                                        job['status'] = "error"
                                        job['progress'] = 0
                                        requests.post(API, json = {"uuid": uuid, "job": job, "type": "update_job"}, headers={"User-Agent": "VOID-Compute-Client"})
                                else:
                                    print(json["message"])
                            else:
                                print("Error: " + str(response))
              else:
                if json["code"] != 404:
                    display(HTML("<font color='red'>" + json["message"] + "</font>"), display_id = "void-error")
        else:
            set_connection_status(uuid, "Waiting for", "orange", "...")
        time.sleep(1)