import numpy as np
from qmlab_testcase import QMLabTest
from qmlab.data_generation import generate_random_data


class TestGenerateData(QMLabTest):
    def setUp(self, num_features: int = 2):
        super().setUp()
        self.num_features = num_features
        self.num_training_examples = 20
        self.num_test_examples = 5

        self.X_train, self.y_train, self.X_test, self.y_test = generate_random_data(
            feature_dimension=self.num_features,
            training_examples_per_class=self.num_training_examples,
            test_examples_per_class=self.num_test_examples,
            random_state=self.random_state,
        )

    def test_shapes(self):
        np.testing.assert_array_equal(self.X_train.shape, (40, self.num_features))
        np.testing.assert_array_equal(self.X_test.shape, (10, self.num_features))
        np.testing.assert_array_equal(self.y_train.shape, (40,))
        np.testing.assert_array_equal(self.y_test.shape, (10,))

    def test_label_values(self):
        np.testing.assert_array_equal(self.y_train, np.hstack(([-1] * 20, [+1] * 20)))
        np.testing.assert_array_equal(
            self.y_test, [-1, -1, -1, -1, -1, +1, +1, +1, +1, +1]
        )
