
class WeightedSample:
    def __init__(self, instances, init=True):
        self.sample = instances
        self.sum = 0

        for instance in self.sample:
            if init:
                instance.weight = 1/len(self.sample)

            self.sum += instance.weight

        self.dist_sum = self.sum

    def normalize(self):
        z = self.dist_sum/self.sum

        for instance in self.sample:
            instance.weight *= z

    def change_weight(self, i, new_weight):
        self.sum -= self.sample[i].weight
        self.sample[i].weight = new_weight
        self.sum += new_weight

    def get(self, i):
        return self.sample[i]

    def size(self):
        return len(self.sample)
