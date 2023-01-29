import datetime
def Log(tag, message):
    now = datetime.datetime.now()
    print(f"[{now.hour}:{now.minute}:{now.second}] {tag}: {message}")