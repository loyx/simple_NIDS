"""
聚类树
"""
import sys
from typing import List
from contextlib import contextmanager
from DataRecord import *


@contextmanager
def redirection(name, mode):
    """ redirect print """
    saved = sys.stdout
    fp = open(name, mode)
    sys.stdout = fp
    yield
    sys.stdout = saved
    fp.close()


def format_msg(padding: str, info='', length=60):
    """ for print """
    if info != '':
        info = ' ' + info + ' '
    return "{:{padding}^{length}}".format(info, padding=padding, length=length)


class ClusterNode:
    """
    聚类树节点
    """

    def __init__(self, str_path=None, parent_node=None):
        self.strPath: str = str_path
        self.parentNode: ClusterNode = parent_node
        self.childNode: List[ClusterNode] = []
        self.labelNum: List[int] = [0] * MAX_LABELS
        self.center: DataNode = DataNode()
        self.clusterResult = 0
        self.isClusterOK = False
        self.isLeaf = 0

    def calCenterDistance(self, record: DataNode):
        """
        计算一条记录与中心的点的距离
        """
        return self.center.EucNorm(record)

    def getChildNode(self, index):
        """
        获取子节点
        """
        return self.childNode[index]

    def getClusterNodeLabel(self):
        """
        获取本聚集类的标签
        """
        return self.center.label

    def getNearestCluster(self, record: DataNode):
        """
        获取与当前数据最相近的聚集类
        return： ClusterNode
        """

        if self.isLeaf > 0:
            return self
        nearest_node = self
        min_distance = self.calCenterDistance(record)
        for child in self.childNode:
            middle_node = child.getNearestCluster(record)
            middle_dist = middle_node.calCenterDistance(record)
            if middle_dist < min_distance:
                nearest_node = middle_node
                min_distance = middle_dist
        return nearest_node

    def print(self):
        """
        聚类树节点输出信息
        """

        """ 本节点信息 """
        print(format_msg(' ', "Cluster: " + self.strPath))
        print(format_msg('-', "Center Information"))
        fmt = "{} = {}"
        for value, name in zip(self.center, ATTRIBUTE_NAMES):
            print(fmt.format(name, value))
        print(fmt.format("label", self.center.label))
        print(format_msg('-', "Cluster Param"))
        print(fmt.format("isClusterOk", self.isClusterOK))
        print(fmt.format("isLeaf", self.isLeaf))
        print(fmt.format("ClusterResult", self.clusterResult))
        print(format_msg('-', "Label Number"))
        for value, name in zip(self.labelNum, LABEL_NAMES):
            print(fmt.format(name, value))
        print(format_msg("="))

        """ 子节点信息 """
        for child in self.childNode:
            child.print()

    def printLog(self):
        """
        输出至log
        """
        with redirection(LOG_FILE, "a"):
            self.print()


class ClusterTree:
    """ 持久化聚类结果 """

    def __init__(self):
        self.rootNode = ClusterNode('0')

    @staticmethod
    def insertNode(parent: ClusterNode, node: ClusterNode):
        node.parentNode = parent
        sub_path = len(parent.childNode)
        parent.childNode.append(node)
        node.strPath = parent.strPath + '.' + str(sub_path)

    def findNearestCluster(self, record: DataNode) -> ClusterNode:
        return self.rootNode.getNearestCluster(record)

    def print(self):
        print("Start printing ClusterTree ...")
        self.rootNode.print()
        print("Printing ClusterTree finished!")

    def printLog(self):
        print("Start printing ClusterTree in log ...")
        with redirection(LOG_FILE, 'a'):
            print("Start printing ClusterTree in log ...")
            self.rootNode.print()
            print("Printing ClusterTree in log finished!")
        print("Printing ClusterTree in log finished!")


if __name__ == '__main__':
    print(format_msg('='))
    print(format_msg('=', "test msg"))
