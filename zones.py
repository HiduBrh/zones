from utils_acc import get_traffic_dataset, plot_2d_geo

dataset = get_traffic_dataset()
labels = dataset[:,2]
plot_2d_geo(dataset[:,0:2])