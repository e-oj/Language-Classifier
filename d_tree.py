import math


def d_tree(examples, features, parent_examples, depth=20):
    """
    Builds the decision tree, selecting features
    based on information gain.

    :param examples: training examples
    :param features: set of features
    :param parent_examples: parent's example set
    :param depth: maximum depth

    :return: The root node of a decision tree
    """
    if not examples:
        return DNode(plurality_value(parent_examples), is_leaf=True)
    if same_goal(examples):
        return DNode(examples[0].goal, is_leaf=True)
    if not features:
        return DNode(plurality_value(examples), is_leaf=True)

    feature, kids = max_gain(examples, features)
    root = DNode(feature)

    if depth < 1:
        depth = 1

    for value in kids:
        exs = kids[value]

        if depth == 1:
            subtree = DNode(plurality_value(exs), is_leaf=True)
            root.add(value, subtree)
        else:
            subtree = d_tree(exs, features.difference({feature}), examples, depth - 1)
            root.add(value, subtree)

    return root


class DNode:
    """
    This class represents a single node in
    a decision tree.
    """

    def __init__(self, value, is_leaf=False):
        """
        Initialize the node.

        :param value: the node's value
        :param is_leaf: is this a leaf node?
        """
        self.is_leaf = is_leaf
        self.value = value
        self.children = {}
        self.weight = None

    def add(self, label, d_node):
        """
        Adds a child to the node.

        :param label: the branch label
        :param d_node: the new node
        """
        self.children[label] = d_node

    def print(self):
        """
        Prints a representation of the node
        and its children.
        """
        out = str(self.value) + " -> "

        for key in self.children:
            out += str(key) + " : " + str(self.children[key].value) + " | "

        print(out)

        for key in self.children:
            self.children[key].print()

    def decide(self, instance):
        """
        Classify an instance of data.

        :param instance: instance of data
        :return: classification
        """
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
    """
    Get the majority classification from a
    node's children

    :param node: the node to be voted on
    :return: majority classification
    """
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


def count_goals(examples):
    """
    Counts the number of examples for
    each classification. Takes weights into
    consideration.

    :param examples: list of examples
    :return: count of every classification
    """
    count = {}

    for ex in examples:
        weight = ex.weight if ex.weight else 1

        if ex.goal in count:
            count[ex.goal] += weight
        else:
            count[ex.goal] = weight

    return count


def plurality_value(examples):
    """
    Gets the majority classification from a
    list of examples.

    :param examples: list of examples.
    :return: majority classification
    """
    value = None
    max_weight = -1
    count = count_goals(examples)

    for ex in examples:
        if count[ex.goal] > max_weight:
            max_weight = count[ex.goal]
            value = ex.goal

    return value


def same_goal(examples):
    """
    Checks if every example in the list
    has the same classification.

    :param examples: list of examples
    :return: True or False
    """
    goal = examples[0].goal

    for i in range(1, len(examples)):
        if examples[i] != goal:
            return False

    return True


def entropy(examples):
    """
    :param examples: list of examples
    :return: entropy of the list
    """
    count = count_goals(examples)
    total = 0

    for key in count.keys():
        p = count[key]/len(examples)
        total += -p * math.log(p, 2)

    return total


def max_gain(examples, features):
    """
    Finds a split of the example list which
    leads to the most information gain.

    :param examples: list of examples
    :param features: set of features

    :return: feature and split with max gain.
    """
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
    """
    Calculates the information gain after
    splitting examples on a feature.

    :param examples: list of examples
    :param feature: feature to split on
    :param entrpy: current entropy

    :return: information gain
    """
    kids = split(examples, feature)
    total = 0

    for kid in kids:
        exs = kids[kid]
        total += (len(exs)/len(examples)) * entropy(exs)

    gains = entrpy - total

    return gains, kids


def split(examples, feature):
    """
    Splits a list of examples on a feature.

    :param examples: list of examples
    :param feature: feature to split on.

    :return: table representing the split.
    """
    result = {}

    for ex in examples:
        value = ex.features[feature]

        if value in result:
            result[value].append(ex)
        else:
            result[value] = [ex]

    return result
