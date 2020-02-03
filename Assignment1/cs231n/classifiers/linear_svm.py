from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i, :]
                dW[:, y[i]] += -X[i, :]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # NUMERICAL way of gradient esitmation, which is super slow and not accurate
    # h = 0.0001                                                                
    # for ii in range(dW.shape[0]):
    #     for jj in range(dW.shape[1]):
    #
    #         W_after = W
    #         W_after[ii, jj] += h
    #         loss_after = 0.0
    #         for i in range(num_train):
    #             scores = X[i].dot(W_after)
    #             correct_class_score = scores[y[i]]
    #             for j in range(num_classes):
    #                 if j == y[i]:
    #                     continue
    #                 margin = scores[j] - correct_class_score + 1 # note delta = 1
    #                 if margin > 0:
    #                     loss_after += margin
    #
    #         # Right now the loss is a sum over all training examples, but we want it
    #         # to be an average instead so we divide by num_train.
    #         loss_after /= num_train
    #
    #         # Add regularization to the loss.
    #         loss_after += reg * np.sum(W * W)
    #
    #         # update dW
    #         dW[ii, jj] = (loss_after - loss) / h
    #############################################################################
    # ANALYTICAL way of gradient calculation, which is fast and accurate
    # Reference: https://math.stackexchange.com/questions/2572318/derivation-of-gradient-of-svm-loss
    # Reference: http://cs231n.github.io/linear-classify/#svm
    # Reference: https://blog.csdn.net/lanchunhui/article/details/70991228
    ### Loss function for i-th sample (TeX):
    # \begin{split}
    # L_i =&\sum_{j\neq y_i}\max(0, s_j-s_{y_i}+1)\\
    # =&\sum_{j\neq y_i} \left[ \max(0, w_j^Tx_i - w_{y_i}^Tx_i + 1) \right]
    # \end{split}
    ### Gradient Derivation (TeX):
    # \begin{split}
    # &\nabla_{w_{y_i}} L_i = - \left( \sum_{j\neq y_i} \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) \right) x_i\\
    # &\nabla_{w_j} L_i = \mathbb{1}(w_j^Tx_i - w_{y_i}^Tx_i + \Delta > 0) x_i\quad j\neq y_i
    # \end{split}
    ### implementation is embeded in the code above
    # TEST: run the following test to check
    # import numpy as np
    # a = np.zeros((3, 2))
    # b = np.random.rand(6)
    # b = np.reshape(b, (2, 3))
    # a[:,0] += b[0,:]
    #############################################################################
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TEST: run the following test code to check
    # a = np.random.randint(10, high=20, size=20) # generate a of size 20, where each element if in [10, 20)
    # a = np.reshape(a, (5, 4))
    # y_index = np.random.randint(4, size=5) # generate y_index of size 5, where each element is up to 4
    # y = np.choose(y_index, a.T) # choose ys from each row of a at y_index
    # margin = np.subtract(a, y[:, np.newaxis]) + 1 # subtract each row with corresponding elements in y
    #############################################################################
    scores = X.dot(W) # 500, 10
    num_train = scores.shape[0] # 500
    correct_class_score = np.choose(y, scores.T) # 500,
    margin = np.subtract(scores, correct_class_score[:, np.newaxis]) + 1
    margin[range(num_train), list(y)] = 0
    loss = np.sum(margin, where=[margin>0]) / num_train + reg * np.sum(W*W)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
