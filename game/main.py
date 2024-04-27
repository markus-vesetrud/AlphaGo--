import torch
import torch.nn as nn
import numpy as np

from parameters import *
from reinforcement_learning import ReinforcementLearning
from parameter_neural_net import LinearResidualNet


# -------------- Hyperparameters -------------
# Search parameters
# Set to None to start from scratch
dataset_path = INITIAL_REPLAY_BUFFER
model_path   = MODEL_START

# --------------------------------------------


# ------------- Other Variables --------------

model = LinearResidualNet(BOARD_SIZE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
if OPTIMIZER == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)
elif OPTIMIZER == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)
elif OPTIMIZER == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)
elif OPTIMIZER == 'rmsprop':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)


if model_path is not None and dataset_path is not None:
    model.load_state_dict(torch.load(model_path))

    with open(dataset_path, 'rb') as f:
        replay_buffer_state: np.ndarray = np.load(f)
        replay_buffer_target: np.ndarray = np.load(f)
else:
    replay_buffer_state = None
    replay_buffer_target = None


# --------------- Main RL loop ---------------


reinforcement_learning = ReinforcementLearning(BOARD_SIZE, EXPLORATION_WEIGHT, EPSILON, EPSILON_DECAY,
                                            NUM_SEARCH, NUM_GAMES, NUM_EPISODES, 
                                            BATCH_SIZE, NUM_EPOCHS, LOG_INTERVAL, SAVE_INTERVAL, loss_fn=criterion, 
                                            optimizer=optimizer, model=model, verbose=True, 
                                            start_epoch=0, replay_buffer_max_length=REPLAY_BUFFER_MAX_LENGTH, 
                                            initial_replay_buffer_state=replay_buffer_state, initial_replay_buffer_target=replay_buffer_target)

reinforcement_learning.main_loop()



