from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # softmax function: L_i = -f_{y_i} + \log\sum_j e^{f_j}
    # analytic gradient reference: https://github.com/Halfish/cs231n/blob/master/assignment1/cs231n/classifiers/softmax.py
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
      scores = X[i].dot(W)
      correct_class_score = scores[y[i]]
      scores_exp_sum = np.sum(np.exp(scores))
      loss += -correct_class_score + np.log(scores_exp_sum)
      dW[:, y[i]] -= X[i]
      for j in range(num_classes):
          dW[:, j] += np.exp(scores[j]) / scores_exp_sum * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    num_train = scores.shape[0]
    correct_class_score = np.choose(y, scores.T)
    scores_exp_sum = np.sum(np.exp(scores), axis=1)
    loss = np.sum(-correct_class_score + np.log(scores_exp_sum)) / num_train + reg * np.sum(W*W)
    pass

    counts = np.exp(scores) / scores_exp_sum.reshape(num_train, 1)
    counts[range(num_train), y] -= 1
    dW = np.dot(X.T, counts) / num_train + reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
