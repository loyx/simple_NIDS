"""
Cluster.py
usage: python Cluster.py [train_data_path test_data_path]
"""
from Kmeans import *


def test_reader(path):
    with open(path, 'r') as fp:
        for line in fp.readlines():
            record = DataNode.fromstr(line.rstrip('\n'))
            yield record


def main():
    try:
        _, train_data_path, test_data_path = sys.argv
    except ValueError:
        train_data_path = 'kddcup.data_10_percent_datatreat'
        test_data_path = 'corrected_datatreat'

    """train"""
    cluster_tree = ClusterTree()
    km = Kmeans(tree=cluster_tree, kid=Kmeans.KMEANS_ID, level=1, num_dimensions=MAX_ATTRIBUTES + 1)
    km.readTrainData(train_data_path)
    k_value = MAX_LABELS

    with redirection(LOG_FILE, 'w'):
        print("Init K-value = ", k_value)

    km.runKmeans(k_value)

    print(format_msg('*', "Total Clustering process finished !"))
    with redirection(LOG_FILE, 'a'):
        print(format_msg('*', "Total Clustering process finished !"))

    cluster_tree.printLog()

    """test"""

    print(format_msg('*', "Start classify the test records"))
    with redirection(LOG_FILE, 'a'):
        print(format_msg('*', "Start classify the test records"))

    reader = test_reader(test_data_path)
    cfs_matrix = ConfuseMatrix()
    right_rcd_mun = 0
    test_rcd_mun = 0
    with redirection(RESULT_FILE, 'w'):
        print(format_msg('*', "Classification result"))
    fmt = "True Label = {} Pre Label = {} Cluster Path = {}"
    for record in reader:
        predict = cluster_tree.findNearestCluster(record)
        if record.label == predict.getClusterNodeLabel():
            right_rcd_mun += 1
        cfs_matrix.update(record.label, predict.getClusterNodeLabel())

        with redirection(RESULT_FILE, 'a'):
            print(fmt.format(LABEL_NAMES[record.label],
                             LABEL_NAMES[predict.getClusterNodeLabel()],
                             predict.strPath))
        test_rcd_mun += 1
        if test_rcd_mun % 10000 == 0:
            print("{} records have been done ...".format(test_rcd_mun))
            with redirection(LOG_FILE, 'a'):
                print("{} records have been done ...".format(test_rcd_mun))

    print(format_msg('*', "The process of classifying test records finished !"))
    with redirection(LOG_FILE, 'a'):
        print(format_msg('*', "The process of classifying test records finished !"))

    print(format_msg('=', "Classify Result"))
    fmt = "Total test record = {} Right label record = {} Right Rate = {}"
    print(fmt.format(test_rcd_mun, right_rcd_mun, right_rcd_mun / test_rcd_mun))
    with redirection(RESULT_FILE, 'a'):
        print(format_msg('=', "Classify Result"))
        print(fmt.format(test_rcd_mun, right_rcd_mun, right_rcd_mun / test_rcd_mun))

    cfs_matrix.print()
    cfs_matrix.printLog()


if __name__ == '__main__':
    main()
