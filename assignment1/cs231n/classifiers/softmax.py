from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
 
  loss = 0.0
  dW = np.zeros(W.shape)
  num_classes=W.shape[1]
  num_train=X[0]
  N=X.shape[0]
  for i in range(N):
    y_hat = X[i] @ W                    # raw scores vector
    y_exp = np.exp(y_hat - y_hat.max()) # numerically stable exponent vector
    softmax = y_exp / y_exp.sum()       # pure softmax for each score
    loss -= np.log(softmax[y[i]])       # append cross-entropy
    softmax[y[i]] -= 1                  # update for gradient
    dW += np.outer(X[i], softmax)       # gradient

    loss = loss / N + reg * np.sum(W**2)    # average loss and regularize 
    dW = dW / N + 2 * reg * W               # finish calculating gradient

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive."""
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N=len(y)
    score=X @ W
    score_exp=np.exp(score-score.max())
    softmax=score_exp/score_exp.sum(axis=1,keepdims=True)
    t=softmax[range(N),y]
    loss-=np.log(t).sum()
    loss=loss/N +reg * np.sum(W**2)
    score_exp[range(N), y] -= 1                  # update P for gradient
    dW = X.T @ score_exp / N + 2 * reg * W       # calculate gradient

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
