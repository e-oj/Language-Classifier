import string

from model import Example
from d_tree import d_tree


def train(examples):
    features = set(examples[0].features.keys())
    tree = d_tree(examples, features, [], 1000)

    return tree


def test(examples, tree):
    correct = 0
    print()

    for ex in examples:
        d = decision(ex, tree)

        if d == ex.goal:
            correct += 1

        print(ex.goal, ex.value[:30], d)

    print()
    print((correct/len(examples)) * 100)


def decision(example, tree):
    node = tree

    while node:
        if node.is_leaf:
            return node.value

        branch = example.features[node.value]

        if branch in node.children:
            node = node.children[branch]
        else:
            print("Sorry! :( ", node.value, branch)
            break

    return None


def main():
    files = ("train.dat", "test.dat")
    lines = ([], [])
    tree = None
    puncs = set(string.punctuation)

    for i in range(len(files)):
        filename = files[i]

        for line in open(filename):
            line = "".join([ch for ch in line if ch not in puncs])
            line = Example(line)

            print(line.goal, ":", line.features, ":", line.value)

            lines[i].append(line)

    if lines[0]:
        tree = train(lines[0])

    if lines[1] and tree:
        test(lines[1], tree)


if __name__ == "__main__":
    main()
