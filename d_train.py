import pickle

from model import Example
from d_tree import d_tree


def train(examples, filename="tree.dt"):
    features = set(examples[0].features.keys())
    tree = d_tree(examples, features, [], 1000)
    f = open(filename, "wb")

    pickle.dump(tree, f)
    f.close()

    return tree


def test(examples, tree):
    correct = 0
    print()

    for ex in examples:
        d = decision(ex, tree)

        if d == ex.goal:
            correct += 1
        else:
            print("oops!")

        print(ex.goal, ex.value[:50], d)

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
            return plurality_value(node)


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
    files = ("train.dat", "test.dat")
    lines = ([], [])
    tree = None

    for i in range(len(files)):
        filename = files[i]

        for line in open(filename):
            if not len(line.strip()) > 3:
                continue

            line = Example(line)

            print(line.goal, ":", line.features, ":", line.value)

            lines[i].append(line)

    if lines[0]:
        tree = train(lines[0])

    if lines[1] and tree:
        test(lines[1], tree)
    #else:
        #f = open("tree.dt", "rb")
        #tree = pickle.load(f)
        #test(lines[1], tree)


if __name__ == "__main__":
    main()
