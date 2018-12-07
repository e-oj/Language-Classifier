import math


def d_tree(examples, features, parent_examples, depth=20):
    if not examples:
        return DNode(plurality_value(parent_examples), True)
    if same_goal(examples):
        return DNode(examples[0].goal, True)
    if not features:
        return DNode(plurality_value(examples), True)

    feature, kids = max_gain(examples, features)
    root = DNode(feature)

    for value in kids:
        exs = kids[value]

        if depth == 0:
            subtree = DNode(plurality_value(exs), True)
            root.add(value, subtree)
        else:
            subtree = d_tree(exs, features.difference({feature}), examples, depth - 1)
            root.add(value, subtree)

    print("depth: ", depth)
    return root


class DNode:
    def __init__(self, value, is_leaf=False):
        self.is_leaf = is_leaf
        self.value = value
        self.children = {}

    def add(self, label, d_node):
        self.children[label] = d_node

    def print(self):
        out = str(self.value) + " -> "

        for key in self.children:
            out += str(key) + " : " + str(self.children[key].value) + ", "

        print(out)

        for key in self.children:
            self.children[key].print()

    def decide(self, instance):
        node = self

        while node:
            if node.is_leaf:
                return node.value

            branch = instance.features[node.value]

            if branch in node.children:
                node = node.children[branch]
            else:
                return vote(node)

        return None


def vote(node):
    if node.is_leaf:
        return node.value

    if not node.children:
        return None

    count = {}
    max_count = -1
    max_val = None

    for branch in node.children:
        val = vote(node.children[branch])

        if not val:
            continue

        count[val] = count[val] + 1 if val in count else 1

        if count[val] > max_count:
            max_count = count[val]
            max_val = val

    return max_val


def plurality_value(examples):
    count = {}
    value = None
    max_count = -1

    for ex in examples:
        if ex.goal in count:
            count[ex.goal] += 1
        else:
            count[ex.goal] = 1

        if count[ex.goal] > max_count:
            max_count = count[ex.goal]
            value = ex.goal

    return value


def same_goal(examples):
    goal = examples[0].goal

    for i in range(1, len(examples)):
        if examples[i] != goal:
            return False

    return True


def entropy(examples):
    count = {}
    total = 0

    for ex in examples:
        if ex.goal in count:
            count[ex.goal] += 1
        else:
            count[ex.goal] = 1

    for key in count.keys():
        p = count[key]/len(examples)
        total += -p * math.log(p, 2)

    return total


def max_gain(examples, features):
    entrpy = entropy(examples)
    max_val = -1
    max_feature = None
    children = None

    for feature in features:
        gains, kids = gain(examples, feature, entrpy)

        if gains > max_val:
            max_val = gains
            max_feature = feature
            children = kids

    return max_feature, children


def gain(examples, feature, entrpy):
    kids = split(examples, feature)
    total = 0

    for kid in kids:
        exs = kids[kid]
        total += (len(exs)/len(examples)) * entropy(exs)

    gains = entrpy - total

    return gains, kids


def split(examples, feature):
    result = {}

    for ex in examples:
        value = ex.features[feature]

        if value in result:
            result[value].append(ex)
        else:
            result[value] = [ex]

    return result
