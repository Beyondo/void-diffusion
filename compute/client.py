import requests, json, threading, time, colab
from IPython.display import display
from IPython.display import HTML
from IPython.display import clear_output;
# import get_ipython()
API = "https://voidops.com/diffusion/api.php"
from IPython import get_ipython
def process_job(job):
    print("Processing job " + job["id"])
    if job['type'] == "run_script":
        try:
            exec(job['script'])
        except Exception as e:
            job['status'] = "error"
            job['progress'] = -1
            print(e)
            return False
    elif job['type'] == "install_vendor":
        try:
            import env
            env.install_vendor()
        except Exception as e:
            job['status'] = "error"
            job['progress'] = -1
            print(e)
            return False
    else:
        job['status'] = "error"
        job['progress'] = -1
        print("Unknown job type")
        return False
    return True
def set_connection_status(uuid, msg, color, end = ""):
    display(HTML("%s <code><font color='%s'>%s</font></code>%s<br><hr><br>" % (msg, color, uuid, end)),  display_id = "void-connection")
def run(uuid):
    clear_output()
    set_connection_status(uuid, "Connecting to", "orange", "...")
    while True:
        response = requests.post(API, json = {"type": "get_jobs", "uuid": uuid}, headers={"User-Agent": "VOID-Compute-Client"})
        if response.status_code == 200:
            if response.text != "":
              r = json.loads(response.text)
              if r["status"] == "ok":
                  set_connection_status(uuid, "Currently working for ", "green")
                  num_jobs = len(r["jobs"])
                  if num_jobs > 0:
                    for job in r["jobs"]:
                        if job['status'] == "pending":
                            # Send a post request to the server
                            job['status'] = "processing"
                            job['progress'] = 0
                            response = requests.post(API, json = {"uuid": uuid, "job": job, "type": "update_job"}, headers={"User-Agent": "VOID-Compute-Client"})
                            if response.status_code == 200:
                                r = json.loads(response.text)
                                if r["status"] == "ok":
                                    if process_job(job):
                                        job['status'] = "complete"
                                        job['progress'] = 100
                                        requests.post(API, json = {"uuid": uuid, "job": job, "type": "update_job"}, headers={"User-Agent": "VOID-Compute-Client"})
                                    else:
                                        job['status'] = "error"
                                        job['progress'] = -1
                                        requests.post(API, json = {"uuid": uuid, "job": job, "type": "update_job"}, headers={"User-Agent": "VOID-Compute-Client"})
                                else:
                                    print(r["message"])
                            else:
                                print("Error: " + str(response))
              else:
                if r["code"] != 404:
                    display(HTML("<font color='red'>" + r["message"] + "</font>"), display_id = "void-error")
        else:
            set_connection_status(uuid, "Waiting for", "orange", "...")
        time.sleep(1)