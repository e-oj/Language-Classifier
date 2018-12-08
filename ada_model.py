import pickle
import math

from parse import parse
from d_tree import d_tree
from weighted_sample import WeightedSample


class AdaModel:
    def __init__(self, train_file="./in/train.dat", test_file="./in/test.dat",
                 out_file="./out/ensemble.dt"):

        files = (train_file, test_file)
        lines = parse(files)

        self.data = {"train": lines[0], "test": lines[1]}
        self.out_file = out_file
        self.ensemble = []
        self.tree = None

    def train(self, ensemble_size=5):
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

    def test(self, h_file=None):
        if not self.ensemble:
            self.train()

        examples = parse([h_file])[0] if h_file else self.data["test"]
        result = []

        for ex in examples:
            d = self.vote(ex)

            result.append({"value": ex.value, "result": d, "goal": ex.goal})

        evaluate(result, examples)

    def vote(self, instance):
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


def evaluate(result, examples):
    correct = 0

    print()

    for res in result:
        if res["result"] == res["goal"]:
            correct += 1
        else:
            print("| oops!")

        print("| value:", res["value"][:50], "| result:", res["result"], "| expected:", res["goal"])

    print()
    print(str((correct / len(examples)) * 100) + "%", "accuracy")


def main():
    model = AdaModel()
    model.train(5)
    model.test()


if __name__ == "__main__":
    main()
