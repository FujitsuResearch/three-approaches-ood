import random
import numpy as np

# make random pair index
def random_pair(y):
    y = np.argmax(y, axis=-1)
    tmp_o_y = np.copy(y)
    pair_idx = []
    for i in range(len(y)):
        o_pair = np.where(tmp_o_y[:,0]==y[i,0])[0]
        ro = random.choice(o_pair)
        pair_idx.append(ro)
        tmp_o_y[ro] = -1
    return pair_idx

def main():
    y = np.array(range(5))
    zeros = np.stack([np.zeros(5),y], 1)
    y = np.concatenate([zeros, zeros+[1,0]],0)
    y = np.concatenate([y, zeros+[2,0]],0)
    y = np.repeat(y, 5, axis=0)
    np.random.shuffle(y)
    pair_idx = random_pair(y)
    for i in range(len(y)):
        print("y:{}, pair_y:{}".format(y[i], y[pair_idx[i]]))

if __name__ == "__main__":
    main()
