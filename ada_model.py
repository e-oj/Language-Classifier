import pickle
import math

from instance import Instance
from parse import parse
from d_tree import d_tree
from weighted_sample import WeightedSample


class AdaModel:
    """
    This class represents a model based on
    adaboost.
    """
    def __init__(self, train_file="./in/train.dat", test_file="./in/test.dat",
                 out_file="./out/ensemble.oj"):
        """
        Initialize the model.

        :param train_file: training data
        :param test_file: test data
        :param out_file: output data
        """
        files = (train_file, test_file)
        lines = parse(files)

        self.data = {"train": lines[0], "test": lines[1]}
        self.out_file = out_file
        self.ensemble = []
        self.tree = None

    def train(self, ensemble_size=5):
        """
        Learns an ensemble using adaboost and
        saves the model to a file.
        """

        examples = self.data["train"]
        features = set(examples[0].features.keys())
        sample = WeightedSample(examples)
        self.ensemble = []

        for i in range(ensemble_size):
            stump = d_tree(examples, features, [], 1)
            error = 0

            for ex in examples:
                decision = stump.decide(ex)

                if decision != ex.goal:
                    error += ex.weight

            for j in range(len(examples)):
                ex = examples[j]
                decision = stump.decide(ex)

                if decision == ex.goal:
                    new_weight = ex.weight * error/(sample.dist_sum - error)
                    sample.change_weight(j, new_weight)

            sample.normalize()
            stump.weight = math.log(sample.dist_sum - error)/error
            self.ensemble.append(stump)

        f = open(self.out_file, "wb")
        pickle.dump(self, f)
        f.close()

    def test(self, test_file=None):
        """
        Tests the model.

        :param test_file: test data
        """
        if not self.ensemble:
            self.train()

        examples = parse([test_file])[0] if test_file else self.data["test"]
        result = []

        for ex in examples:
            d = self.vote(ex)

            result.append({"value": ex.value, "result": d, "goal": ex.goal})

        evaluate(result, examples)

    def predict(self, line):
        """
        Predicts a single line

        :param line: line to test
        :return: prediction (en or nl)
        """
        if not self.ensemble:
            self.train()

        ex = Instance(line, preserve=True)
        d = self.vote(ex)
        print("| value:", ex.value, "| result:", d)
        return d

    def vote(self, instance):
        """
        Classifies an instance by collecting
        votes from the ensemble

        :param instance: instance to classify
        :return: classification
        """
        count = {}
        max_count = 0
        winner = None

        for stump in self.ensemble:
            decision = stump.decide(instance)

            if decision in count:
                count[decision] += stump.weight
            else:
                count[decision] = stump.weight

            if count[decision] > max_count:
                max_count = count[decision]
                winner = decision

        return winner


def evaluate(results, examples):
    """
    Evaluates results from a model.

    :param results: list of results
    :param examples: test data
    """
    correct = 0

    print()

    for res in results:
        if res["result"] == res["goal"]:
            correct += 1
        else:
            print("| oops!")

        print("| value:", res["value"][:50], "| result:", res["result"], "| expected:", res["goal"])

    print()
    print(str((correct / len(examples)) * 100) + "%", "accuracy")


def main():
    """
    Main function. (Test)
    """
    model = AdaModel()
    model.train(5)
    # model.predict("en|as another store room. The ramp has been discreetly slipped into the screened area at")
    # model.predict("en|means that each team will try to get their first four (at least) riders across")
    # model.predict("nl|Zijn eerste boek The Peasants Revolt verscheen in 2009. Zijn tweede boek over het Huis")
    model.test()


if __name__ == "__main__":
    main()
