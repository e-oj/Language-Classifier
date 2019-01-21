import pickle

from instance import Instance
from parse import parse
from d_tree import d_tree


class DecisionModel:
    """
    This class represents a model based on
    decision trees.
    """

    def __init__(self, train_file="./in/train.dat", test_file="./in/test.dat",
                 out_file="./out/tree.oj"):
        """
        Initialize the model

        :param train_file: training data
        :param test_file: test data
        :param out_file: output data
        """
        files = (train_file, test_file)
        lines = parse(files)

        self.data = {"train": lines[0], "test": lines[1]}
        self.out_file = out_file
        self.tree = None

    def train(self):
        """
        Learns a decision tree and saves
        the model to a file.
        """
        examples = self.data["train"]
        features = set(examples[0].features.keys())

        self.tree = d_tree(examples, features, [], 7)

        f = open(self.out_file, "wb")
        pickle.dump(self, f)
        f.close()

    def test(self, test_file=None):
        """
        Tests the model.

        :param test_file: test data
        """
        if not self.tree:
            self.train()

        examples = parse([test_file])[0] if test_file else self.data["test"]
        result = []

        for ex in examples:
            d = self.tree.decide(ex)

            result.append({"value": ex.value, "result": d, "goal": ex.goal})

        evaluate(result, examples)

    def predict(self, line):
        """
        Predicts a single line

        :param line: line to test
        :return: prediction (en or nl)
        """
        if not self.tree:
            self.train()

        ex = Instance(line, preserve=True)
        d = self.tree.decide(ex)
        print("| value:", ex.value, "| result:", d)
        return d

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
    model = DecisionModel()
    model.train()
    # model.predict("as another store room. The ramp has been discreetly slipped into the screened area at")
    # model.predict("stelde zich ten doel om zoveel mogelijk van het verleden van Arundel te redden en")
    model.test()


if __name__ == "__main__":
    main()
