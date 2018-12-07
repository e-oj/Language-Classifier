class WeightedSample:
    def __init__(self, instances):
        self.data = instances
        self.sum = 0

        for instance in self.data:
            instance.weight = 1
            self.sum += instance.weight

        self.dist_sum = self.sum

    def normalize(self):
        z = self.dist_sum/self.sum
        self.sum = 0

        for instance in self.data:
            instance.weight *= z
            self.sum += instance.weight

    def change_weight(self, i, new_weight):
        self.sum -= self.data[i].weight
        self.data[i].weight = new_weight
        self.sum += new_weight

    def get(self, i):
        return self.data[i]

    def size(self):
        return len(self.data)
