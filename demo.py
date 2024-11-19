from qmlab.preprocessing import parse_biomed_data_to_ndarray, downsample_biomed_data
from qmlab.plotting import labels_pie_chart
import matplotlib.pyplot as plt

X, y, feature_names = parse_biomed_data_to_ndarray("sobar_new", return_X_y=True)
X, y = downsample_biomed_data(X, y)
fig = labels_pie_chart(y, title="balanced now")
plt.show()
