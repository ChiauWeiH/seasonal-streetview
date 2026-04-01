import os  
import matplotlib.pyplot as plt  
  
def show_train_history(train_history, loss, accuracy,file_name):
    plt.plot(train_history.history[loss])
    plt.plot(train_history.history[accuracy])
    plt.title('Train History')
    plt.ylabel('')
    plt.xlabel('Epoch')  
    plt.legend(['loss', 'acc'], loc='upper left')
    plt.savefig(file_name + '.png')
    plt.show()
