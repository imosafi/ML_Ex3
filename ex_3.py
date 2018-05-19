import os
import sys
import numpy as np
import random
import pickle
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


SAVE_DATA = False
DATA_PATH = 'data'
EPOCHS = 70
INPUT_DIM = 784
OUTPUT_DIM = 10
BATCH_SIZE = 20
# learning_rate = 0.01
learning_rate = 0.003
epsilon = 0.0001


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    # numerically stable softmax calculation
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def save_data_np():
    train_x = np.loadtxt(os.path.join(DATA_PATH, 'train_x'))
    train_y = np.loadtxt(os.path.join(DATA_PATH, 'train_y'))

    np.save(os.path.join(DATA_PATH, 'train_x'), train_x / 255)
    np.save(os.path.join(DATA_PATH, 'train_y'), train_y )


def get_model_output(input, W, bw, U, bu):
    # l1_output = np.tanh(np.dot(input, W) + bw)
    # l1_output = relu(np.dot(input, W) + bw)
    l1_output = sigmoid(np.dot(input, W) + bw)
    l2_output = np.dot(l1_output, U) + bu
    probs = softmax(l2_output)
    return probs


def predict(input, W, bw, U, bu):
    probs = get_model_output(input, W, bw, U, bu)
    return np.argmax(probs)


def predict_test_set(path, W, bw, U, bu):
    predictions = []
    test_x = np.loadtxt(path) / 255
    for l in test_x:
        predictions.append(str(predict(l, W, bw, U, bu)) + '\n')

    with open('test.pred', 'w+') as f:
        f.writelines(predictions)


def calc_weight_matrix_grad(x, grad_b):
    return np.array([x]).transpose().dot([grad_b])


def calc_grads(x, y, W, bw, U, bu):

    y_label = np.zeros(bu.shape)
    y_label[y] = 1

    y_pred = get_model_output(x, W, bw, U, bu)
    loss = -np.log(y_pred[y])

    z = np.array(x).dot(W) + bw
    # a = np.tanh(z)
    # a = relu(z)
    a = sigmoid(z)
    bu_grad = y_pred - y_label
    U_grad = calc_weight_matrix_grad(a, bu_grad)
    # bw_grad = bu_grad.dot(U.transpose()) * (1 - np.power(a, 2))
    # bw_grad = bu_grad.dot(U.transpose()) * relu_derivative(a)
    bw_grad = bu_grad.dot(U.transpose()) * (a * (1 - a))
    W_grad = calc_weight_matrix_grad(x, bw_grad)

    return loss, W_grad, bw_grad, U_grad, bu_grad


def print_labels_count(all_y):
    arr = [0] * 10
    for y in all_y:
        arr[int(y)] += 1
    for i in xrange(10):
        print 'label {}: {}'.format(i, arr[i])


def split_into_train_val(all_x, all_y):
    c = list(zip(all_x, all_y))
    random.shuffle(c)
    all_x, all_y = zip(*c)
    train_x, train_y, val_x, val_y = [], [], [], []
    each_label_count = int((0.8 * len(all_x)) / 10)
    train_labels_count = [0] * 10
    for x,y in zip(all_x, all_y):
        y = int(y)
        if train_labels_count[y] < each_label_count:
            train_x.append(x)
            train_y.append(y)
            train_labels_count[y] += 1
        else:
            val_x.append(x)
            val_y.append(y)

    return train_x, train_y, val_x, val_y


def init_weights(input_dim, hidden_dim, output_dim):
    W = np.random.uniform(-0.1, 0.1, (input_dim, hidden_dim))
    U = np.random.uniform(-0.1, 0.1, (hidden_dim, output_dim))
    bw = np.random.uniform(-0.1, 0.1, (hidden_dim))
    bu = np.random.uniform(-0.1, 0.1, (output_dim))

    return W, bw, U, bu


def calc_numeric_gradients(x, y, W, bw, U, bu):
    apx_gW = np.zeros(np.shape(W))
    apx_gbw = np.zeros(np.shape(bw))
    apx_gU = np.zeros(np.shape(U))
    apx_gbu = np.zeros(np.shape(bu))

    for i in xrange(W.shape[0]):
        for j in xrange(W.shape[1]):
            W[i][j] += epsilon

            y_pred = get_model_output(x, W, bw, U, bu)
            j_plus = -np.log(y_pred[y])
            W[i][j] -= 2 * epsilon
            y_pred = get_model_output(x, W, bw, U, bu)
            j_minus = -np.log(y_pred[y])
            W[i][j] += epsilon
            apx_gW[i][j] = (j_plus - j_minus)/(2.0 * epsilon)

    for i in xrange(U.shape[0]):
        for j in xrange(U.shape[1]):
            U[i][j] += epsilon

            y_pred = get_model_output(x, W, bw, U, bu)
            j_plus = -np.log(y_pred[y])
            U[i][j] -= 2 * epsilon
            y_pred = get_model_output(x, W, bw, U, bu)
            j_minus = -np.log(y_pred[y])
            U[i][j] += epsilon
            apx_gU[i][j] = (j_plus - j_minus)/(2.0 * epsilon)

    for i in xrange(bw.shape[0]):
        bw[i] += epsilon
        y_pred = get_model_output(x, W, bw, U, bu)
        j_plus = -np.log(y_pred[y])
        bw[i] -= 2 * epsilon
        y_pred = get_model_output(x, W, bw, U, bu)
        j_minus = -np.log(y_pred[y])
        bw[i] += epsilon
        apx_gbw[i] = (j_plus - j_minus)/(2.0 * epsilon)

    for i in xrange(bu.shape[0]):
        bu[i] += epsilon
        y_pred = get_model_output(x, W, bw, U, bu)
        j_plus = -np.log(y_pred[y])
        bu[i] -= 2 * epsilon
        y_pred = get_model_output(x, W, bw, U, bu)
        j_minus = -np.log(y_pred[y])
        bu[i] += epsilon
        apx_gbu[i] = (j_plus - j_minus)/(2.0 * epsilon)

    return apx_gW, apx_gbw, apx_gU, apx_gbu


def print_confusion_matrix(val_x, val_y, W, bw, U, bu):
    confusion_martix = []
    for i in xrange(bu.shape[0]):
        confusion_martix.append([0] * 10)

    for x, y in zip(val_x, val_y):
        confusion_martix[y][predict(x, W, bw, U, bu)] += 1

    print('print confustion matrix:')
    for l in xrange(bu.shape[0]):
        print(confusion_martix[l])


def save_train_val_acc_loss(train_acc, val_acc, train_loss, val_loss):
    with open(os.path.join(DATA_PATH, 'results.txt'), 'w') as f:
        for i in train_acc:
            f.write('{},'.format(i))
        f.write('\n')

        for i in val_acc:
            f.write('{},'.format(i))
        f.write('\n')

        for i in train_loss:
            f.write('{},'.format(i))
        f.write('\n')

        for i in val_loss:
            f.write('{},'.format(i))
        f.write('\n')


def create_train_loss_plots(dir):
    with open(os.path.join(DATA_PATH, 'results.txt'), 'r') as f:
        lines = []
        for l in f.readlines():
            lines.append(map(float, l.strip().split(',')[:-1]))

        plt.title('Net Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        blue_patch = mpatches.Patch(color='blue', label='training')
        red_patch = mpatches.Patch(color='red', label='validation')
        plt.plot([i for i in xrange(len(lines[0]))], lines[0], color='tab:blue')
        plt.plot([i for i in xrange(len(lines[1]))], lines[1], color='tab:red')
        plt.legend(handles=[red_patch, blue_patch], loc=4)

        plt.savefig('Accuracies.png', dpi=100)

        plt.clf()

        plt.title('Net Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        blue_patch = mpatches.Patch(color='blue', label='training')
        red_patch = mpatches.Patch(color='red', label='validation')
        plt.plot([i for i in xrange(len(lines[2]))], lines[2], color='tab:blue')
        plt.plot([i for i in xrange(len(lines[3]))], lines[3], color='tab:red')
        plt.legend(handles=[red_patch, blue_patch], loc=1)

        plt.savefig('Losses.png', dpi=100)


def relu(x):
    return np.maximum(x, 0, x)

def relu_derivative(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def main():
    if True:
        save_data_np()
    all_x = np.load(os.path.join(DATA_PATH, 'train_x.npy'), 'r')
    all_y = np.load(os.path.join(DATA_PATH, 'train_y.npy'), 'r')

    print 'labels count: '
    print_labels_count(all_y)

    train_x, train_y, val_x, val_y = split_into_train_val(all_x, all_y)

    print 'train labels:'
    print_labels_count(train_y)
    print 'val labels:'
    print_labels_count(val_y)

    W, bw, U, bu = init_weights(784, 100, 10)

    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    best_W, best_bw, best_U, best_bu, best_val_acc, best_val_loss = 0, 0, 0, 0, 0.0, 99999.0
    for i in xrange(EPOCHS):
        c = list(zip(train_x, train_y))
        random.shuffle(c)
        train_x, train_y = zip(*c)
        train_accuracy = 0.0
        train_total_loss = 0.0
        sum_W_grad, sum_bw_grad, sum_U_grad, sum_bu_grad = 0, 0, 0, 0

        print('starting epoch {}'.format(i+1))
        k = 0
        for x, y in zip(train_x, train_y):
            if predict(x, W, bw, U, bu) == int(y):
                train_accuracy += 1
            loss, W_grad, bw_grad, U_grad, bu_grad = calc_grads(x, int(y), W, bw, U, bu)


            train_total_loss += loss
            sum_W_grad -= learning_rate * W_grad
            sum_bw_grad -= learning_rate * bw_grad
            sum_U_grad -= learning_rate * U_grad
            sum_bu_grad -= learning_rate * bu_grad


            if ((k != 0 and k % 15 == 0) or k == len(train_x) - 1):
                W = W + sum_W_grad
                bw = bw + sum_bw_grad
                U = U + sum_U_grad
                bu = bu + sum_bu_grad
                sum_W_grad, sum_bw_grad, sum_U_grad, sum_bu_grad = 0, 0, 0, 0

            k += 1

        train_loss.append(str(train_total_loss / len(train_x)))
        train_acc.append(str(train_accuracy / len(train_x)))
        print('train loss: {}'.format(train_total_loss / len(train_x)))
        print('train accuracy: {}'.format(train_accuracy / len(train_x)))

        val_total_loss = 0.0
        val_accuracy = 0.0
        for x, y in zip(val_x, val_y):
            y_distribution = get_model_output(x, W, bw, U, bu)
            val_total_loss += -np.log(y_distribution[y])
            if np.argmax(y_distribution) == y:
                val_accuracy += 1
        val_loss.append(str(val_total_loss / len(val_x)))
        val_acc.append(str(val_accuracy / len(val_x)))
        print('validation loss: {}'.format(val_total_loss / len(val_x)))
        print('validation accuracy: {}'.format(val_accuracy / len(val_x)))
        if float(val_acc[-1]) > best_val_acc and float(val_loss[-1]) < best_val_loss:
            best_val_acc = float(val_acc[-1])
            best_val_loss = float(val_loss[-1])
            best_W = np.copy(W)
            best_bw = np.copy(bw)
            best_U = np.copy(U)
            best_bu = np.copy(bu)

    print('best val accuracy {}'.format(best_val_acc))
    print('best val loss {}'.format(best_val_loss))
    save_train_val_acc_loss(train_acc, val_acc, train_loss, val_loss)
    create_train_loss_plots(os.path.join(DATA_PATH, 'results.txt'))
    print_confusion_matrix(val_x, val_y, best_W, best_bw, best_U, best_bu)
    predict_test_set(os.path.join(DATA_PATH, 'test_x'), best_W, best_bw, best_U, best_bu)


if __name__ == '__main__':
    main()
