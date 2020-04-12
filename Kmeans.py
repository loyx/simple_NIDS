"""
Kmeans主程序及相关类
"""
from itertools import takewhile
from copy import deepcopy
from DataRecord import *
from ClusterTree import *


class ConfuseMatrix:
    """
    混淆矩阵
    """

    def __init__(self):
        self.matrix = [[0] * MAX_LABELS for _ in range(MAX_LABELS)]

    def update(self, true_label, false_label):
        self.matrix[true_label][false_label] += 1

    def __str__(self):
        fmt = "{:{align}10}"
        msg = fmt.format("T/P", align='^')
        for label in LABEL_NAMES:
            msg += fmt.format(label, align='^')
        for index, row in enumerate(self.matrix):
            msg += '\n'
            msg += fmt.format(LABEL_NAMES[index], align='^')
            for value in row:
                msg += fmt.format(value, align='^')
        return msg

    def print(self):
        print(format_msg("Confusion Matrix"))
        print(self)

    def printLog(self):
        with redirection(RESULT_FILE, 'a'):
            self.print()


class Cluster:
    """
    聚类结构
    """

    def __init__(self, center: DataNode):
        self.center: DataNode = deepcopy(center)
        self.memberList: List[DataNode] = [self.center]
        self.numMembers = 1


class Kmeans:
    """
    算法主程序
    """
    KMEANS_ID = 0

    def __init__(self, tree: ClusterTree, kid, level, num_dimensions,
                 data_list=None, self_node=None):
        self.inFile = None

        """ 训练数据集 """
        if data_list is None:
            data_list = []
        self.recordList: List[DataNode] = data_list  # 不创建副本

        """ garbage """
        self.numClusters = 0
        self.numRecord = 0
        self.numDimension = num_dimensions

        """ 算法执行过程中的数据结构 """
        self.clusters: List[Cluster] = []
        self.clusterLevel = level
        self.kmeansID = kid

        """ 保存聚类结果的数据结构 """
        self.clusterTree: ClusterTree = deepcopy(tree)
        self.clusterNodes: List[ClusterNode] = []
        if self_node is None:
            self_node = tree.rootNode
        self.selfClusterNode: ClusterNode = self_node

    def readTrainData(self, path):
        """
        read train data
        """
        count = 0
        print("Start reading Records from training file ...")
        with open(path, 'r') as fp:
            for line in fp.readlines():
                self.recordList.append(DataNode.fromstr(line.rstrip('\n')))
                count += 1
                if count % 100000 == 0:
                    print(format_msg('-', str(count) + " lines have read"))
            self.numRecord = count
        print("Records have read from training file!")

    def initCluster(self, num_clusters):
        """
        任意选择k个初始聚类中心点
        """
        for record in self.recordList:
            for has_found_cluster in self.clusters:
                if has_found_cluster.center == record:
                    break
            else:
                self.clusters.append(Cluster(record))
                num_clusters -= 1
                if num_clusters == 0:
                    break
        else:
            print("Warning! The number of categories is less than the set value!")
        self.numClusters = len(self.clusters)

    def distributeSample(self):
        """
        将数据对象分配给距离最近的的聚类
        """
        for cluster in self.clusters:
            cluster.memberList.clear()
            cluster.numMembers = 0

        for record in self.recordList:
            closet_cluster = self.findClosetCluster(record)
            closet_cluster.memberList.append(record)
            closet_cluster.numMembers += 1

    def calNewClusterCenter(self):
        """
        重新计算聚类中心点
        """
        same_as_before = []
        for cluster in self.clusters:
            """
            每个中心点前三个数据取众数，其余取平均数，label无意义
            """
            new_center = DataNode()

            """计算平均"""
            counter = [dict()] * 3
            for record in cluster.memberList:
                for index in range(3):
                    try:
                        counter[index][record[index]] += 1
                    except KeyError:
                        counter[index][record[index]] = 1
                new_center += record
            new_center /= cluster.numMembers

            """更新前三个属性"""
            for index in range(3):
                max_value = 0
                max_key = None
                for key, value in counter[index].items():
                    if value > max_value:
                        max_value = value
                        max_key = key
                new_center[index] = max_key

            """判断是否和前中心点一样，不考虑label"""
            same_as_before.append(all(a == b for a, b in zip(cluster.center, new_center)))

            cluster.center = new_center
        return all(same_as_before)

    def findClosetCluster(self, record: DataNode) -> Cluster:
        """
        找到与给定数据对象最近的一个聚类
        """
        centers = list(map(lambda x: x.center, self.clusters))
        dists = list(map(record.EucNorm, centers))
        return self.clusters[dists.index(min(dists))]

    def printCluster(self):
        """
        print
        """
        with redirection(LOG_FILE, 'a'):
            print(format_msg('='))
            print(format_msg(' ', "There are totally {} clusters.".format(self.numClusters)))
            fmt = "{} = {}"
            for index, cluster in enumerate(self.clusters):
                print(format_msg('*', "Cluster Center {}".format(index)))
                for name, value in zip(ATTRIBUTE_NAMES, cluster.center):
                    print(fmt.format(name, value))
                print(format_msg('-'))
                print("Number of members in Cluster {} = {}".format(index, len(cluster.memberList)))

    def runKmeans(self, k_value):
        """
        kmeans 算法过程
        """

        is_finish = False
        circle_num = 0

        print("Start clustering process!")
        with redirection(LOG_FILE, 'a'):
            print("Start clustering process!")

        """初始化聚类"""
        self.initCluster(k_value)
        while not is_finish:
            circle_num += 1

            print(format_msg('-', "Circle " + str(circle_num)))
            with redirection(LOG_FILE, 'a'):
                print(format_msg('-', "Circle " + str(circle_num)))

            """聚类第二步， 分配数据对象"""
            self.distributeSample()
            """第三步， 重新计算聚类中心点"""
            is_finish = self.calNewClusterCenter()

        print("The process of clustering is finished !")
        print("Print clustering result ...")
        with redirection(LOG_FILE, 'a'):
            print("The process of clustering is finished !")
            print("Print clustering result ...")

        """创建对应的ClusterTree，持久化保存聚类结果"""
        self.createClusterTreeNode(self.selfClusterNode)
        """打印聚类效果"""
        self.getClustersLabel()

        """递归聚类"""
        for index, cluster in enumerate(self.clusters):

            print("Check Kmeans:{} Cluster:{}".format(self.kmeansID, index))
            with redirection(LOG_FILE, 'a'):
                print("Check Kmeans:{} Cluster:{}".format(self.kmeansID, index))

            if self.isClusterOK(index):

                fmt = "Level: {} Kmeans: {} cluster: {} OK!"
                print(fmt.format(self.clusterLevel, self.kmeansID, index))
                print(format_msg('-'))
                with redirection(LOG_FILE, 'a'):
                    print(fmt.format(self.clusterLevel, self.kmeansID, index))
                    print(format_msg('-'))

            else:
                self.printClusterLabel(index)
                next_k_value = self.getDiffLabelOfCluster(index)

                fmt = "Level: {} Kmeans: {} cluster: {} need go on!"
                print(fmt.format(self.clusterLevel, self.kmeansID, index))
                print(format_msg('-', length=40))
                print("Set k = ", next_k_value)
                with redirection(LOG_FILE, 'a'):
                    print(fmt.format(self.clusterLevel, self.kmeansID, index))
                    print(format_msg('-', length=40))
                    print("Set k = ", next_k_value)

                next_train_data = self.getClusterList(index)
                next_level = self.clusterLevel + 1
                Kmeans.KMEANS_ID += 1
                next_kmeans = Kmeans(self.clusterTree, self.KMEANS_ID, next_level,
                                     MAX_ATTRIBUTES + 1, next_train_data,
                                     self.clusterNodes[index])
                next_kmeans.runKmeans(next_k_value)

                print(format_msg('-', length=40))
                print("Stop recursion ! Go back !")
                with redirection(LOG_FILE, 'a'):
                    print(format_msg('-', length=40))
                    print("Stop recursion ! Go back !")

    @staticmethod
    def countClusterLabel(cluster: Cluster):
        """
        获取一个cluster的label结果
        """
        label_counter = [0] * MAX_LABELS
        for record in cluster.memberList:
            label_counter[record.label] += 1
        return label_counter, label_counter.index(max(label_counter))

    def getClustersLabel(self):
        """
        聚类结束后，查询所有聚类标签数
        持久化聚类结果
        """

        print(format_msg('*', "Cluster Level" + str(self.clusterLevel)))
        with redirection(LOG_FILE, 'a'):
            print(format_msg('*', "Cluster Level" + str(self.clusterLevel)))

        for cluster_id, cluster, cluster_node in enumerate(zip(self.clusters, self.clusterNodes)):

            print('-', "Keams:{} Cluster:{}".format(self.kmeansID, cluster_id))
            with redirection(LOG_FILE, 'a'):
                print('-', "Keams:{} Cluster:{}".format(self.kmeansID, cluster_id))

            label_counter, max_label = self.countClusterLabel(cluster)
            for index, label_num in enumerate(label_counter):
                cluster_node.labelNum[index] = label_num
            cluster_node.center.label = max_label

            fmt = "{} = {}"
            for name, value in zip(LABEL_NAMES, label_counter):
                print(fmt.format(name, value))
                with redirection(LOG_FILE, 'a'):
                    print(fmt.format(name, value))
        print('*')
        with redirection(LOG_FILE, 'a'):
            print('*')

    def getPrecisionAndOtherLabel(self, cluster: Cluster):
        label_counter, max_label = self.countClusterLabel(cluster)
        other_label_num = sum(label_counter) - label_counter[max_label]
        precision = other_label_num / label_counter[max_label]
        return precision, other_label_num

    def isClusterOK(self, index):
        """
        检查聚类后cluster的结果是否合理
        """
        cluster = self.clusters[index]
        cluster_node = self.clusterNodes[index]

        precision, other_label_num = self.getPrecisionAndOtherLabel(cluster)
        cluster_node.clusterResult = precision

        """ 打印相信息，设置对应ClusterNode"""
        def setTrueCondition(criterion, st, ed):
            print(st, " <= ", criterion, " < ", ed)
            with redirection(LOG_FILE, 'a'):
                print(st, " <= ", criterion, " < ", ed)
            cluster_node.isClusterOK = True
            cluster_node.isLeaf = 1

        def setFalseCondition(criterion, ed):
            print(criterion, ' > ', ed)
            with redirection(LOG_FILE, 'a'):
                print(criterion, ' > ', ed)
            cluster_node.isClusterOK = False
            cluster_node.isLeaf = 0

        """
        根据当前聚集类的层、杂质label个数和聚类精度判断是否聚类合理
        """
        if self.clusterLevel <= INNER_LEVEL:  # 当前聚类层数小于3
            if other_label_num > 100:  # 层数小， 杂质label大于100，直接要求再聚类
                setFalseCondition(other_label_num, 100)
                return False
            elif precision < CLUSTER_PRECISION:
                setTrueCondition(precision, 0, CLUSTER_PRECISION)
                return True
            else:
                setFalseCondition(precision, CLUSTER_PRECISION)
                return False
        elif self.clusterLevel <= MAX_LEVEL:  # 聚类层数大于3小于8

            diff_level = self.clusterLevel - INNER_LEVEL

            if other_label_num > 500:
                setFalseCondition(other_label_num, 500)
                return False
            elif precision < diff_level * CLUSTER_PRECISION:
                c = CLUSTER_PRECISION
                internal = takewhile(lambda x: x*c < precision <= x*c+c, range(1, diff_level+1))
                internal = next(internal)
                start, end = internal*c, (internal+1)*c
                setTrueCondition(precision, start, end)
                return True
            else:
                setFalseCondition(precision, diff_level*CLUSTER_PRECISION)
                return False
        else:  # 聚类层数过深异常结束
            self.selfClusterNode.isLeaf = 2
            self.selfClusterNode.isClusterOK = True

    def getClusterList(self, index):
        """
        获取聚类对应的数据集
        """
        return self.clusters[index].memberList

    def printClusterLabel(self, index):
        """
        打印一个聚类中的不同label
        """
        label_count, _ = self.countClusterLabel(self.clusters[index])

        fmt = "{} = {}"
        print(format_msg(' ', "Cluster " + self.clusterLevel + '.' + index))
        print(format_msg('-', length=30))
        for name, value in zip(LABEL_NAMES, label_count):
            print(fmt.format(name, value))

        with redirection(LOG_FILE, 'a'):
            print(format_msg(' ', "Cluster " + self.clusterLevel + '.' + index))
            print(format_msg('-', length=30))
            for name, value in zip(LABEL_NAMES, label_count):
                print(fmt.format(name, value))

    def getDiffLabelOfCluster(self, index):
        """
        获取一个聚类中不同label的个数
        """
        label_count, _ = self.countClusterLabel(self.clusters[index])
        return sum(map(lambda x: 1 if x else 0, label_count))

    def createClusterTreeNode(self, parent: ClusterNode):
        """
        为聚类结果创建节点
        """
        assert len(self.clusterNodes) == 0
        for cluster in self.clusters:
            cluster_node = ClusterNode()
            cluster_node.center = deepcopy(cluster)
            self.clusterNodes.append(cluster_node)
            self.clusterTree.insertNode(parent, cluster_node)


if __name__ == '__main__':
    mt = ConfuseMatrix()
    print(mt)
