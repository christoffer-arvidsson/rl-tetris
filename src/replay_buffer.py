import numpy as np
import random

from tree import SumSegmentTree, MinSegmentTree

class ReplayBuffer(object):
    """ Cyclic replay buffer. """
    def __init__(self, capacity):
        self.capacity = capacity
        self.storage = []
        self.current = 0

    def __len__(self):
        return len(self.storage)

    def push(self, element):
        if len(self.storage) < self.capacity:
            self.storage.append(None)
        self.storage[self.current] = element
        self.current = (self.current + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.storage, batch_size)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def push(self, *args, **kwargs):
        idx = self.current
        super().push(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self.storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        samples = list(map(self.storage.__getitem__, idxes))

        return samples, weights, idxes

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
