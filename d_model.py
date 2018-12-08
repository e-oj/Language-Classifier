import pickle

from parse import parse
from d_tree import d_tree


class DecisionModel:
    def __init__(self, train_file="./in/train.dat", test_file="./in/test.dat",
                 out_file="./out/tree.dt"):
        files = (train_file, test_file)
        lines = parse(files)

        self.data = {"train": lines[0], "test": lines[1]}
        self.out_file = out_file
        self.tree = None

    def train(self):
        examples = self.data["train"]
        features = set(examples[0].features.keys())

        self.tree = d_tree(examples, features, [], 6)

        f = open(self.out_file, "wb")
        pickle.dump(self, f)
        f.close()

    def test(self, h_file=None):
        if not self.tree:
            self.train()

        examples = parse([h_file])[0] if h_file else self.data["test"]
        result = []

        for ex in examples:
            d = self.tree.decide(ex)

            result.append({"value": ex.value, "result": d, "goal": ex.goal})

        evaluate(result, examples)


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
    model = DecisionModel()
    model.train()
    model.test()


if __name__ == "__main__":
    main()
