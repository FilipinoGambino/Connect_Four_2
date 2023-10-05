from abc import ABC, abstractmethod

class Action(ABC):
    def __init__(self, act_type: str) -> None:
        self.act_type = act_type
        self.repeat = 0

    def __str__(self):
        print("Action Template")

    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError("")

class PlaceTokenAction(Action):
    def __init__(self, position: int) -> None:
        super().__init__(f"position_{position}")
        self.position = position

    def __str__(self) -> str:
        return f"{self.act_type}"

    def __call__(self, **kwargs):
        return self.position