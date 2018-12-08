from instance import Instance


def parse(files):
    """
    Parses instances from a file

    :param files: a collection of files
    :return: a collection of instances
    """
    lines = [[] for _ in files]

    for i in range(len(files)):
        filename = files[i]

        for line in open(filename):
            if not len(line.strip()) > 3:
                continue

            lines[i].append(Instance(line))

    return lines
