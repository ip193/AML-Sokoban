import threading
import time

def do_first():
    time.sleep(10)
    print("I was called first")

def do_second():
    print("I was called second")

class FirstThread(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        do_first()

class SecondThread(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        do_second()


threads = [FirstThread(), SecondThread()]

for t in threads:
    t.start()


