class WeightedSample:
    """
    This class represents a weighted sample.
    The weights of every instance in the sample
    form a distribution.
    """

    def __init__(self, instances):
        """
        Initialize the sample with a list of
        instances.

        :param instances: list of instances.
        """
        self.data = instances
        self.sum = 0

        for instance in self.data:
            instance.weight = 1
            self.sum += instance.weight

        self.dist_sum = self.sum

    def normalize(self):
        """
        Make the weights conform to the
        distribution
        """
        z = self.dist_sum/self.sum
        self.sum = 0

        for instance in self.data:
            instance.weight *= z
            self.sum += instance.weight

    def change_weight(self, i, new_weight):
        """
        Change the weight of an instance in
        the sample

        :param i: index of the instance
        :param new_weight: new weight
        """
        self.sum -= self.data[i].weight
        self.data[i].weight = new_weight
        self.sum += new_weight

    def size(self):
        """
        :return: Sample size
        """
        return len(self.data)
