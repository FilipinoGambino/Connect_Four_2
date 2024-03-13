class Player:
    def __init__(self, player_id):
        self.mark = player_id
        self.reward = 0.

    def __str__(self):
        return f"Player {self.mark}"
