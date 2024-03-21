import threading
import time
import multiprocessing as mp

class MyClass:
    def __init__(self):
        self.x = 0

    def increment(self):
        self.x += 1

def thread1(my_class, free_queue, full_queue):
    while True:
        index = free_queue.get()
        if index is None:
            break
        my_class.increment()
        full_queue.put(index)
        time.sleep(.1)

def thread2(my_class, free_queue, full_queue):
    while True:
        index = full_queue.get()
        if index is None:
            break
        my_class.increment()
        free_queue.put(index)
        time.sleep(.1)

if __name__ == '__main__':
    def test():
        free_queue = mp.SimpleQueue()
        full_queue = mp.SimpleQueue()

        free_queue.put(0)

        my_class = MyClass()

        t1 = threading.Thread(target=thread1, args=(my_class,free_queue,full_queue,))
        t2 = threading.Thread(target=thread2, args=(my_class,free_queue,full_queue,))

        t1.start()
        t2.start()
        try:
            while my_class.x < 10:
                print(my_class.x)
        except KeyboardInterrupt:
            return
        finally:
            free_queue.put(None)
            full_queue.put(None)
            t1.join(timeout=1)
            t2.join(timeout=1)
        print(my_class.x)
    test()