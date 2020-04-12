"""
聚类数据结构及一些全局变量
"""
import reprlib
import numbers
from array import array


LOG_FILE = "Log.txt"
RESULT_FILE = "Result.txt"

INNER_LEVEL = 3
MAX_LEVEL = 8
CLUSTER_PRECISION = 0.1

ATTRIBUTE_NAMES = [
    "ProtocolType",
    "Service",
    "StatusFlag",
    "SrcBytes",
    "DesBytes",
    "FailedLogins",
    "NumOfRoot",
    "Count",
    "SrvCount",
    "RerrorRate",
    "SameSrvRate",
    "DiffSrvRate",
    "DstHostSrvCount",
    "DstHostSameSrvRate",
    "DstHostDiffSrvRate",
    "DstHostSameSrcPortRate",
    "DstHostSrvDiffHostRate",
    "DstHostSrvSerrorRate",
]
MAX_ATTRIBUTES = len(ATTRIBUTE_NAMES)

LABEL_NAMES = [
    "normal",
    "dos",
    "probe",
    "u2r",
    "r2l",
]
MAX_LABELS = len(LABEL_NAMES)


class DataNode:
    """
    聚类数据节点
    """

    typecode = 'f'

    __slots__ = ["_attributes", "label"]

    def __init__(self, iterable=None, label=-1):
        if iterable is None:
            iterable = [0] * MAX_ATTRIBUTES
        if len(iterable) > MAX_ATTRIBUTES:
            raise ValueError("Attributes should less than {}".format(MAX_ATTRIBUTES))
        self._attributes = array(self.typecode, iterable)
        self.label: int = label

    def __len__(self):
        return len(self._attributes)

    def __getitem__(self, item):
        return self._attributes[item]

    def __getattr__(self, item):
        cls = type(self)
        try:
            index = ATTRIBUTE_NAMES.index(item)
        except ValueError:
            msg = "{.__name__!r} object has no attribute {!r}"
            raise AttributeError(msg.format(cls, item))
        else:
            return self._attributes[index]

    def __setitem__(self, key, value):
        self._attributes[key] = value

    def __iadd__(self, other):
        if not isinstance(other, DataNode):
            try:
                iterable = iter(other)
            except TypeError:
                self_cls = type(self).__name__
                msg = "right operand in += must be {!r} or an iterable"
                raise TypeError(msg.format(self_cls))
            other = list(iterable)
        for index, value in enumerate(other):
            self[index] += value
        return self

    def __itruediv__(self, other):
        if not isinstance(other, numbers.Real):
            self_cls = type(self).__name__
            msg = "right operand in /= must be a number"
            raise TypeError(msg.format(self_cls))
        for index in range(3, MAX_ATTRIBUTES):
            self._attributes[index] /= other
        return self

    def __eq__(self, other):
        """
        label相同或数值完全一样
        """
        return self.label == other.label or all(a == b for a, b in zip(self, other))

    def __repr__(self):
        attr = reprlib.repr(self._attributes)
        attr = attr[attr.find('['):-1]
        return "DR(attr:{}, label:{})".format(attr, self.label)

    def EucNorm(self, other):
        """
        计算两个点间的欧式距离
        """

        dist = 0
        gen_other = iter(other)
        gen_self = iter(self)

        """
        对于前三个category属性，相同距离为0，不同距离为1 
        c_weight表示前三个属性的权重
        """
        c_weight = 1
        cnt = 0
        while cnt < 3:
            dist += c_weight*int(next(gen_other) == next(gen_self))
            cnt += 1

        """ 其余属性计算欧式距离 """
        dist += sum(map(lambda x, y: (x-y) ** 2, gen_other, gen_self))

        return dist

    @classmethod
    def fromstr(cls, str_record: str):
        """
        parse data from str
        """
        data = str_record.split(',')
        attributes = list(map(float, data[:-1]))
        assert len(attributes) == MAX_ATTRIBUTES
        label = int(data[-1])
        return cls(attributes, label)


if __name__ == '__main__':
    test = "2,7,6,181,5450,0,0,0,0,8,8,0,100,0,9,100,0,11,0,0,0"
    dr = DataNode.fromstr(test)
    print(dr)
    print(dr.Count)
