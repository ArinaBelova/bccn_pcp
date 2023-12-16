Connect 4 game (solved game), turn-based, perfect information

Goal: 4 pieces in the row


Course: minimax/MCTS agent (by the end of week 6)

Project: another algorithm


Tournament


30 Oct 2023: code for playing the game (algorithm)
Minimax and MCTS


# Questions
 
- Q learning: definition of works/does not work
- States??? 
- When is hard deadline? 
- How to take care of invalid actions, is it be baked inside of the DQN training??
- Can we do separate training of DQN and separate run of the game logics? 
I cannot imaging training on the go, how do we get actions of the other party?
So we play with minimax! Or do we?
- Self-play: https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Training-Configuration-File.md#self-play

Move 2 is invalid: Selected column is full.
Player 1 lost by making an illegal move.

- How do we set rewards? 
  - Lke minimax: loose (-inf) /win (+inf)
  - Extention: Slightly punish for the number of moves (??)
  - Extention: experience buffer

- Extention: CNN before MLP (in the very end)  

- The training happens only inside of the agent_dqn
