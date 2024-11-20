import pytest
import sklearn
from qmlab.kernel.angle_embedded_kernel import QSVC
from qmlab_testcase import QMLabTest


class TestQSVC(QMLabTest):

    def test_qsvc_is_abstract(self):
        with pytest.raises(TypeError):
            qsvm = QSVC()
