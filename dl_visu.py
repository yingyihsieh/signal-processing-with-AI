import matplotlib.pyplot as plt
import numpy as np

def acc_line(epochs,train_acc,test_acc):
    plt.figure()
    plt.plot(np.arange(1, epochs + 1, 1), train_acc, 'r',
             np.arange(1, epochs + 1, 1), test_acc, 'b')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('avg accuracy per epochs')
    plt.show()

def loss_line(epochs,train_loss,test_loss):
    plt.figure()
    plt.plot(np.arange(1, epochs + 1, 1), train_loss, 'r',
             np.arange(1, epochs + 1, 1), test_loss, 'b')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('avg loss per epochs')
    plt.show()