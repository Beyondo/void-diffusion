requirements = [ 'transformers' ]
import time
def run(args, callback):
    while True:
        time.sleep(1)
        callback(0, args['seeds'][0] + "Waiting...")
        time.sleep(1)
        callback(50, args['seeds'][0] + "Running...")
        time.sleep(1)
        callback(100, args['seeds'][0] + "Complete!")
    return True