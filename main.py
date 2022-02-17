from typing import Tuple, Iterable

import numpy as np

from dataset import load_dataset
from neural_net import NeuralNet

from tqdm import tqdm, trange

def get_batch(inputs:Iterable, targets:Iterable, batch_size:int, shuffle:bool=True) -> Tuple[Iterable]:
    """ Ud fra hele datasættet lav batches med inputs og targets af batch_size størrelse """

    if shuffle:
        indices = np.random.permutation(len(inputs))

    for start_idx in tqdm(range(0, len(inputs) - batch_size + 1, batch_size), position=1, desc="batch", leave=False, colour="Cyan", ncols=80):
        if shuffle:
            data_range = indices[start_idx:start_idx + batch_size]
        else:
            data_range = slice(start_idx, start_idx + batch_size)

        yield inputs[data_range], targets[data_range]

# Load dataen
x_train, y_train, x_val, y_val, X_test, y_test = load_dataset(flatten=True)

# Initialiser det neurale netværk
neural_net = NeuralNet(
    inputs_len = x_train.shape[1],
    hidden_len = 8,
    output_len = 10,
    learning_rate = [0.1, 0.1]
)

def train_fn(inputs:Iterable, targets:Iterable, val_inputs:Iterable, val_targets:Iterable) -> int:
    """ 
    Her trænes det neural netværk på en epoch
    
    hvor inputs, targets er dataen vi træner på
    og val_inputs og val_targets er test sættet
    """
    
    for x_batch, y_batch in get_batch(inputs, targets, batch_size = 32, shuffle=True):
        # Beregn feedforward netværket
        neural_net_memory = neural_net.feedforward(x_batch)
        logits = neural_net_memory[-1]

        # Backpropagate igennem netværket
        loss = neural_net.backprop(logits, y_batch, neural_net_memory)

    # Test netværkets nuværende performance
    predictions = neural_net.predict(val_inputs)
    return np.mean(predictions.all() == y_batch.all())

# Gem alle accuracies her
accuracy = []

# Træn hele netværket på antal epochs
epochs = 20
for e in tqdm(range(epochs), position=0, desc="epoch", leave=False, colour="Green", ncols=80):
    acc = train_fn(x_train, y_train, x_val, y_val)
    accuracy.append(acc)
    
# Import matplotlib for at kunne plotte accuracy grafen
import matplotlib.pyplot as plt

# Tilføj graf af testsættets accuracy
plt.plot(accuracy,label='val accuracy')

plt.legend(loc='best')
plt.grid()
plt.show()
