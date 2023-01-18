import threading, os, importlib, time
from compute import job
importlib.reload(job)
jobManagerThread = None
job_queue = [] # List of job objects
threads = []
finished = False
max_num_parallel_jobs = 3
running_jobs = []
def queue_thread():
    global job_queue, finished, threads
    while True:
        # Run a maximum of 3 threads at a time
        # Wait for threads to finish before starting new ones
        num_is_alive = sum([thread.is_alive() for thread in threads])
        if num_is_alive < max_num_parallel_jobs:
            if len(job_queue) > 0:
                threads.append(threading.Thread(target=job_queue.pop(0).process))
                threads[-1].start()
            elif finished:
                for thread in threads:
                    if thread.is_alive():
                        thread.join()
                break
        time.sleep(1)
def run():
    global jobManagerThread, finished
    jobManagerThread = threading.Thread(target=queue_thread)
    finished = False
    jobManagerThread.start()

def add_to_queue(job):
    global job_queue, running_jobs
    for existingJob in job_queue:
        if existingJob.data['id'] == job.data['id']:
            break
    print("Adding to queue %s (Currently running: %s)" % (job.data['id'], len(running_jobs)))
    job.data['status'] = "In Queue"
    job.data['progress'] = 0
    job.update()
    job_queue.append(job)

def join_queue_thread():
    global jobManagerThread, finished
    finished = True
    jobManagerThread.join()


def contains(uuid, job_id):
    for job in job_queue:
        if job.uuid == uuid and job.data.id == job_id:
            return True
    return False

def signal_termination(job_id):
    for job in job_queue:
        if job.data['id'] == job_id:
            job_queue.remove(job)
            job.stop()
            return True
    for job in running_jobs:
        if job.data['id'] == job_id:
            running_jobs.remove(job)
            job.stop()
            return True
    return False