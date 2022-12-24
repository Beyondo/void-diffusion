import requests, json, threading, time, colab
from IPython.display import display
from IPython.display import HTML
from IPython.display import clear_output;
# import get_ipython()
from IPython import get_ipython
def job(data):
    pass
def set_connection_status(uuid, msg, color, end = ""):
    display(HTML("%s <code><font color='%s'>%s</font></code>%s<br><hr><br>" % (msg, color, uuid, end)),  display_id = "void-connection")
def run(uuid):
    clear_output()
    set_connection_status(uuid, "Connecting to", "orange", "...")
    while True:
        response = requests.post("https://voidops.com/diffusion/api.php", json = {"type": "get_jobs", "uuid": uuid}, headers={"User-Agent": "VOID-Compute-Client"})
        if response.status_code == 200:
            if response.text != "":
              data = json.loads(response.text)
              if data["status"] == "ok":
                  set_connection_status(uuid, "Currently working for ", "green")
                  num_jobs = len(data["jobs"])
                  if num_jobs > 0:
                    print("Processing %d jobs..." % num_jobs)
                    for job in data["jobs"]:
                        # Send a post request to the server
                        job['progress'] = 0
                        response = requests.post("https://voidops.com/diffusion/api.php", json = {"uuid": uuid, "job": job, "type": "update_job"}, headers={"User-Agent": "VOID-Compute-Client"})
                        if response.status_code == 200:
                            data = json.loads(response.text)
                            if data["status"] == "ok":
                                print(data["message"])
                            else:
                                print(data["message"])
                        else:
                            print("Error: " + str(response))
              else:
                if data["code"] != 404:
                    display(HTML("<font color='red'>" + data["message"] + "</font>"), display_id = "void-error")
        else:
            set_connection_status(uuid, "Waiting for", "orange", "...")
        time.sleep(1)