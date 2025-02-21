
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


filenames = pickle.load(open('data/filenames-caltech101.pickle', 'rb'))
feature_list = pickle.load(open('data/features-caltech101-resnet.pickle', 'rb'))


from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute',
metric='euclidean').fit(feature_list)


distances, indices = neighbors.kneighbors([feature_list[0]])



plt.imshow(mpimg.imread(filenames[indices[0][3]]))
plt.show()