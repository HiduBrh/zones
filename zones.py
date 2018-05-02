from utils_acc import get_traffic_dataset,sorted_eigen_values_and_vectors, plot_2d_geo

dataset = get_traffic_dataset()
values, vectors= sorted_eigen_values_and_vectors(dataset)
labels = dataset[:,2]
plot_2d_geo(dataset[:,0:2])