from abc import ABC, abstractmethod
from functools import update_wrapper
import logging
import time
from types import GeneratorType, SimpleNamespace

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
        "[Line number: %(lineno)-3s ] " "%(message)s"
    ),
    level=20,
)

def decorator_one(func):
    start = time.time()
    def inside(*args):
        output = func(*args)

        if isinstance(output, list):
            return {x:y for x,y in zip(reversed(output), output)}
        elif isinstance(output, GeneratorType): # Apparently you can't use the keyword 'generator' to check type
            y,*output = output
            for x in output:
                y *= x
            return y
        else:
            return output
    logging.info(f"{str(func):^55} took {time.time() - start:.5f} seconds to run")

    return inside

class DecoratorTwo(object):
    def __init__(self, func):
        self.func = func
        '''
        update_wrapper(self, func)
        For the life of me, I can't figure out how to get this class decorator to see and adjust instance
        variables like changing self.string to Console on line 96. Maybe you'll figure it out! Maybe it's not even
        possible, idk.
        '''

    def __call__(self, *args, **kwargs):
        start = time.time()
        time.sleep(1)
        self.func(self, *args, **kwargs)
        logging.info(f"{str(self.func):^55} took {time.time() - start:.5f} seconds to run")

class Computer(ABC):
    def __init__(self):
        self.string = f"Hi! My name is {self.__class__.__name__}  | My instance hash is {self.__hash__()}"

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def update(self, *args):
        pass

class Desktop(Computer):
    def __init__(self):
        '''
        self.string doesn't exist in any Desktop instance until you run super().__init__()
        Also, if you hover over 'Desktop', this entire string will show up.
        '''
        super(Desktop, self).__init__()
        # self.storage_sizes = {"small": 1024, "medium": 2048, "large": 4096}
        self.storage_sizes = {"small": 4096, "medium": 1024, "large": 2048}

    def read(self):
        return self.string

    @decorator_one
    def update(self, string):
        self.string = string + f" with instance hash {self.__hash__()}"


class Laptop(Computer):
    def __init__(self):
        super(Laptop, self).__init__()
        self.storage_sizes = {"small": 256, "medium": 512, "large": 1024}

    def read(self):
        return self.string

    @DecoratorTwo
    def update(self, *args):
        self.string = f"Hi! My name is Console | My instance hash is {self.__hash__()}"

@decorator_one
def generator(numbers):
    for x in numbers:
        yield x

@decorator_one
def list_comprehension(numbers):
    return [x for x in numbers]

if __name__=='__main__':
    import yaml
    with open('config.yaml','r') as config:
        flags = yaml.safe_load(config)

    logging.info(flags)
    logging.info(flags['disable_output'])

    flags = SimpleNamespace(**flags)
    if flags.disable_output:
        logging.disable(level=20)

    desktop = Desktop()
    laptop = Laptop()
    logging.info(desktop.read())
    logging.info(laptop.read())
    desktop.update("This is a class called 'Manually Input Name'")
    laptop.update()
    logging.info(desktop.read())
    logging.info(laptop.read())

    logging.info(f"items: {[value for value in desktop.storage_sizes.items()]}")
    logging.info(f"keys: {[value for value in desktop.storage_sizes.keys()]}")
    logging.info(f"values: {[value for value in desktop.storage_sizes.values()]}")
    logging.info(f"sorted by key: {[value for value in sorted(desktop.storage_sizes.items(), key=lambda x: x[0])]}")
    logging.info(f"sorted by value: {[value for value in sorted(desktop.storage_sizes.items(), key=lambda x: x[1])]}")
    logging.info([f"{key1}: {val1} | {key2}: {val2}" for (key1,val1),(key2,val2) in zip(desktop.storage_sizes.items(), laptop.storage_sizes.items())])

    logging.info(f"{generator([*range(1,10)]):e}")
    logging.info(list_comprehension([*range(5)])) # I think I told you this was a generator. My b