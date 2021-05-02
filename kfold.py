import random
import numpy as np


class KFold(object):
    def __init__(self, files=None, idx=None, labels=None, k=1, random_state=None, shuffle=True):
        assert k > 1, print('K must greater than 1.')
        self.k = k
        if files:
            self.idx = range(len(files))
        elif idx:
            self.idx = idx
        else:
            assert False, print('No index found.')

        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)

        self.labels = labels
        self.random_state = random_state
        self.shuffle = shuffle

        if self.labels:
            if isinstance(self.labels, list):
                self.labels = np.array(self.labels)
            # num_classes
            nc = len(set(self.labels))
            assert np.sum(self.labels >= nc) == 0, print("The label do not match the num_classes.")

            class_bins = []
            for i in range(nc):
                class_idx = np.where(self.labels == i)[0]
                if len(class_idx) < k:
                    class_idx = np.repeat(class_idx, repeats=(k / len(class_idx) + 1))
                if self.shuffle:
                    np.random.shuffle(class_idx)

                class_bin = []
                step = int(len(class_idx) / k)
                for j in range(k-1):
                    class_bin.append(class_idx[step*j: step*(j+1)].tolist())
                class_bin.append(class_idx[step*(k-1):].tolist())
                class_bins.append(class_bin)

            split_res = []
            # count = []
            for i in range(k):
                sp = []
                for j in range(nc):
                    sp += class_bins[j][(j+i) % k]
                split_res.append(sp)
                # count += sp

            # assert len(count) == len(set(count)), print('Duplicate in index')
            self.split = split_res

    def _get_split_iter(self):
        iters = []
        for i in range(self.k):
            val = self.split[i]
            train = []
            for j in range(self.k):
                if j == i:
                    continue
                train += self.split[j]
            iters.append([train, val])

        return iters


if __name__ == '__main__':
    index = list(range(40))  # (0 ~ 39)
    labels = random.choices(range(4), k=38)  # 38
    labels += [3, 3]


    kf = KFold(idx=index, labels=labels, k=3)

    for t, v in kf._get_split_iter():
        print(t, v)