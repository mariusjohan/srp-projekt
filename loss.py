import numpy as np
from numpy import arange, log, sum, exp, zeros_like

def softmax_crossentropy_with_logits(logits, targets):
    logits_for_answers = logits[arange(len(logits)), targets]
    return log(sum(exp(logits), axis=-1)) - logits_for_answers

def grad_softmax_crossentropy_with_logits(logits, targets):
    ones_for_answers = zeros_like(logits)
    ones_for_answers[arange(len(logits)), targets] = 1

    softmax = exp(logits) / exp(logits).sum(axis=-1, keepdims=True)

    return (softmax - ones_for_answers) / logits.shape[0]