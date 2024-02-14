class Player:
    def __init__(self, player_id):
        self.mark = player_id
        self.active = True if self.mark == 1 else False

    def __str__(self):
        return f"Player {self.mark}"

    def active(self, turn):
        return self.mark == (turn % 2 + 1)