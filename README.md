The purpose of this project was to understand how professional machine learning engineers structure their code
and as a comprehension exercise on the winner of LuxAI season 1 hosted on Kaggle. 

I worked my way through it and had to edit a great deal to get it to behave correctly as a single agent playing 
Connect Four. Of course, it's a bit overkill, but it works, though it seemed to peak after only a few hundred 
thousand game steps and failed to consistently outmatch the Negamax algorithm. The MCTS algorithm was superior
and after writing one up, I submitted it to the ongoing Kaggle compeition where it reached 43 out of 193 teams.
To get higher, it would seem my agent would have to go first because didn't seem to be able to effectively take
advantage of misplays as player two.

https://github.com/FilipinoGambino/MCTS-Connect-Four
