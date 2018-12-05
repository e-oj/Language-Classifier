import pickle

from parse import parse
from d_tree import d_tree


class DecisionModel:
    def __init__(self, train_file, test_file, out_file):
        files = (train_file, test_file)
        lines = parse(files)

        self.data = {"train": lines[0], "test": lines[1]}
        self.out_file = out_file
        self.tree = None

    def train(self):
        examples = self.data["train"]
        features = set(examples[0].features.keys())

        self.tree = d_tree(examples, features, [], 1000)

        f = open(self.out_file, "wb")
        pickle.dump(self.tree, f)
        f.close()

    def test(self):
        if not self.tree:
            self.train()

        examples = self.data["test"]
        result = []

        for ex in examples:
            d = decision(ex, self.tree)

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


def decision(example, tree):
    node = tree

    while node:
        if node.is_leaf:
            return node.value

        branch = example.features[node.value]

        if branch in node.children:
            node = node.children[branch]
        else:
            return plurality_value(node)

    return None


def plurality_value(node):
    if node.is_leaf:
        return node.value

    if not node.children:
        return None

    count = {}
    max_count = -1
    max_val = None

    for branch in node.children:
        val = plurality_value(node.children[branch])

        if not val:
            continue

        if val in count:
            count[val] += 1
        else:
            count[val] = 1

        if count[val] > max_count:
            max_count = count[val]
            max_val = val

    return max_val


def main():
    model = DecisionModel("train.dat", "test.dat", "tree.dt")
    model.train()
    model.test()


if __name__ == "__main__":
    main()
