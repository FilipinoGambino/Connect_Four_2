class Player:
    def __init__(self, player_id):
        self.mark = player_id

    def __str__(self):
        return f"Player {self.mark}"

    def active(self, turn):
        return self.mark == (turn % 2 + 1)


players = [Player(1), Player(2)]
for step in range(43):
    for player in players:
        activity = "ACTIVE" if player.active(step) else "INACTIVE"
        print(f"{player} is {activity} on step {step}")
