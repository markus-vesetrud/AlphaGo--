# AlphaGo--


## TODO

- [x] Make an interface for game logic
- [x] Make basic game Nim, using the interface, to be used for debugging purposes
- [x] Make board representation, update functions, and visualisation for Hex
- [x] Test Hex till we are happy
- [x] Make a general MCTS system with rollouts

- [x] Make a neural network that may take game state as input and gives an action as output
- [x] Rotate board 180 degrees for doubling training instances from data
- [x] Invert board on red's turns to maximize training data
- [ ] Test adam optimizer with 128 batch size
- [x] Maybe apply softmax to mcts output
- [x] Use the network in MCTS
- [x] Change default policy to be epsilon greedy
- [x] Deal with the ocilating behaviour by adding a replay buffer
- [ ] Check epsilon greedy VS proportional output
- [ ] Check no board flipping
- [ ] Check different way of swapping out replay buffer

## Dependencies

numpy, matplotlib, pytorch

