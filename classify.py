import sys
import pickle

from d_model import DecisionModel
from ada_model import AdaModel


def train(examples, out_file, learner):
    if learner == "dt":
        model = DecisionModel(train_file=examples, out_file=out_file)
    else:
        model = AdaModel(train_file=examples, out_file=out_file)

    model.train()


def predict(h_file, test_file):
    h_file = open(h_file, "rb")
    model = pickle.load(h_file)

    h_file.close()
    model.test(test_file)


def main():
    action = sys.argv[1]

    if action == "train":
        examples = sys.argv[2]
        out_file = sys.argv[3]
        learner = sys.argv[4]

        print("Training...")
        train(examples, out_file, learner)
        print("Done.")

    elif action == "predict":
        h_file = sys.argv[2]
        test_file = sys.argv[3]

        predict(h_file, test_file)


if __name__ == '__main__':
    main()
