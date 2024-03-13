from abc import ABC, abstractmethod
from functools import update_wrapper
import logging
import time
from types import SimpleNamespace

'''
logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)
'''
logging.basicConfig(
    format=(
        "[Line number: %(lineno)d ] " "%(message)s"
    ),
    level=20,
)

def decorator_one(func):
    def inside(*args):
        return func(*args)
    return 1 / inside()

class DecoratorTwo(object):
    def __init__(self, func):
        self.func = func
        update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        start = time.time()
        self.func(self, *args, **kwargs)
        logging.info(f"{self.__class__.__name__} took {time.time() - start} seconds to update")
        return

class Computer(ABC):
    class DecoratorThree(object):
        def __init__(self, func):
            self.func = func
            update_wrapper(self, func)

        def __call__(self, *args, **kwargs):
            start = time.time()
            self.func(self, *args, **kwargs)
            time.sleep(1)
            logging.info(f"{self.__class__.__name__} took {time.time() - start} seconds to update")
            return

    def __init__(self):
        self.string = f"This is a class called {self.__class__.__name__}"

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def update(self, *args):
        pass

class Desktop(Computer):
    def __init__(self):
        super(Desktop, self).__init__()
        # self.storage_sizes = {"small": 1024, "medium": 2048, "large": 4096}
        self.storage_sizes = {"small": 4096, "medium": 1024, "large": 2048}

    def read(self):
        logging.info(self.string)

    @DecoratorTwo
    def update(self, string):
        self.string = string
        return


class Laptop(Computer):
    def __init__(self):
        super(Laptop, self).__init__()
        self.storage_sizes = {"small": 256, "medium": 512, "large": 1024}

    def read(self):
        logging.info(self.string)

    @Computer.DecoratorThree
    def update(self, *args):
        self.string = f"This is a class called something stupid probably"
        return

@decorator_one
def another_func():
    return 3

if __name__=='__main__':
    import yaml
    with open('config.yaml','r') as config:
        flags = yaml.safe_load(config)

    print(flags)
    print(flags['disable_output'])
    flags = SimpleNamespace(**flags)
    if flags.disable_output:
        logging.disable(level=20)

    d = Desktop()
    l = Laptop()
    d.read()
    l.read()
    d.update("This is a class called 'Manually Input Name'")
    l.update()
    d.read()
    l.read()
    logging.info([value for value in d.storage_sizes.items()])
    logging.info([value for value in d.storage_sizes.keys()])
    logging.info([value for value in d.storage_sizes.values()])
    logging.info([value for value in sorted(d.storage_sizes.items(), key=lambda x: x[1])])
    logging.info([f"{key1}: {val1} | {key2}: {val2}" for (key1,val1),(key2,val2) in zip(d.storage_sizes.items(), l.storage_sizes.items())])
    logging.info([f"{key1}: {val1} | {key2}: {val2}" for (key1,val1),(key2,val2) in zip(d.storage_sizes.items(), l.storage_sizes.items())])

    logging.info(another_func)