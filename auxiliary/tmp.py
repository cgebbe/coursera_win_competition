import numpy as np


def log_loss(y_true, y_pred):
    mask_true = y_true == 1
    loss = 0
    loss += -np.sum(np.log(np.clip(y_pred[mask_true], 1E-10, 1 - 1E-10)))
    loss += -np.sum(np.log(np.clip(1 - y_pred[~mask_true], 1E-10, 1 - 1E-10)))
    return loss


def main():
    y_true = np.zeros(4)
    list_y_pred = [np.ones(4) * 0.5,
                   np.array([4, 5, 5, 6]) / 10.0,
                   np.array([0, 0, 0, 1]),
                   ]
    for y_pred in list_y_pred:
        loss = log_loss(y_true, y_pred)
        print("For y_pred={}, loss={}".format(y_pred, loss))


if __name__ == '__main__':
    main()
