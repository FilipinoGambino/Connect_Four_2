from abc import ABC, abstractmethod
import logging

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
    level=0,
)


class DecoratorOne:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        output = self.func(args)
        return output ** 2

def decorator_two(func):
    def inside(*args):
        return func(args)
    return inside()

class Computer(ABC):
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
        super(Desktop).__init__()
        self.storage_sizes = {"small": 1024, "medium": 2048, "large": 4096}

    def read(self):
        logging.info(self.string)

    def update(self, string):
        self.string = string


class Laptop(Computer):
    def __init__(self):
        super(Laptop).__init__()
        self.storage_sizes = {"small": 256, "medium": 512, "large": 1024}

    def read(self):
        logging.info(self.string)

    def update(self):
        self.string = f"This is a class called {self.__class__.__name__}"

if __name__=='__main__':
    d = Desktop()
    l = Laptop()
    d.read()
    l.read()
    d.update("This is a class called 'Manually Input Name'")
    l.update()
    d.read()
    l.read()