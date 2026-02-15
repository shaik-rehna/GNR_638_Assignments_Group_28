import random
import my_framework as mf


class DataLoader:

    def __init__(self, dataset, batch_size=32, shuffle=True):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = list(range(len(dataset)))

    def __iter__(self):

        if self.shuffle:
            random.shuffle(self.indices)

        for start in range(0, len(self.indices), self.batch_size):

            batch_indices = self.indices[start:start + self.batch_size]

            images = []
            labels = []

            for idx in batch_indices:
                img, label = self.dataset[idx]
                images.append(img)
                labels.append(label)

            yield self._stack(images), labels

    def _stack(self, tensors):

        batch_size = len(tensors)

        # shape of single image
        c, h, w = tensors[0].shape

        stacked_data = []

        for t in tensors:
            stacked_data.extend(t.data)

        return mf.Tensor(
            stacked_data,
            [batch_size, c, h, w],
            False
        )