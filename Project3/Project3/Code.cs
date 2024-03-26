using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace Project3
{
    internal class Code
    {

        //import matplotlib.pyplot as plt
        //import seaborn as sns
        //from bs4 import BeautifulSoup
        //results = soup.find(id="ResultsContainer")

        /*
         for item in jobElementsResults:
  titleEl = item.find("h2",class_="title")
  companyEl = item.find("h3", class_="subtitle")
  locationEl = item.find("p", class_="location")
  print(title_element)
  print(company_element)
  print(location_element)
  print()
         */

        //        from sklearn.datasets import fetch_openml
        //df = fetch_openml('titanic', version = 1, as_frame = True)['data']

        //df.dropna(subset = ['age'], inplace = True)
        //df.select_dtypes('category').head()
        //df['parch'] = df['parch'].astype(int)

        //        from sklearn.impute import SimpleImputer
        //df = fetch_openml('titanic', version = 1, as_frame = True)['data']

        //imp = SimpleImputer(strategy = 'median')
        //df['age'] = imp.fit_transform(df[['age']])
        //df['age'].isnull().sum()
        //df['age'].mean()

        //from sklearn.datasets import load_iris
        //from sklearn import tree

        //clf = tree.DecisionTreeClassifier().fit(X,y)
        //from sklearn.model_selection import train_test_split
        //tree.plot_tree(clf)


        //        from sklearn.datasets import make_classification
        //from sklearn.preprocessing import StandardScaler
        //import matplotlib.pyplot as plt
        //X, y = make_classification(n_samples = 100, n_features = 2, n_informative = 2, n_redundant = 0, n_clusters_per_class = 2, random_state = 42)
        //        scaler = StandardScaler()
        //X_scaled = scaler.fit_transform(X)

        //plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c= y, cmap= 'viridis', marker = '*')
        //plt.show()

        //        from sklearn import tree
        //clf = tree.DecisionTreeClassifier()
        //clf = clf.fit(X_train, y_train)


        //        from sklearn.preprocessing import StandardScaler
        //from sklearn.pipeline import make_pipeline

        //from mlxtend.evaluate import confusion_matrix

        //y_pred = clf.predict(X_test)

        //confmat = confusion_matrix(y_test, y_pred)

        //print(confmat)
        //            from mlxtend.plotting import plot_confusion_matrix
        //import matplotlib.pyplot as plt

        //fig, ax = plot_confusion_matrix(conf_mat = confmat, figsize = (2, 2))
        //plt.show()

        //        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        //print("ACC", accuracy_score(y_true= y_test, y_pred= y_pred))
        //print("Precision", precision_score(y_true= y_test, y_pred= y_pred))
        //print("Recall", recall_score(y_true= y_test, y_pred= y_pred))
        //print("F1 score", f1_score(y_true= y_test, y_pred= y_pred))

        //        from sklearn.metrics import classification_report
        //print(classification_report(y_test, y_pred))

        //        from sklearn.linear_model import LinearRegression
        //        x_reshaped = x.reshape(-1, 1)
        //reg = LinearRegression().fit(x_reshaped, y_with_noise)
        //reg.score(x_reshaped, y_with_noise)

        //from sklearn.neighbors import KNeighborsClassifier
        //        knn = KNeighborsClassifier(n_neighbors= 3)
        //knn.fit(X, y)
        //y_new = knn.predict(X[-1].reshape(1, -1))
        //y_new


        //    def eucledian_distance(a, b):
        //distance = 0.0
        //if(len(a) != len(b)):
        //  print("a and b don't have the same size ")
        //else:
        //  for i in range(len(a)) :
        //    distance += (a[i]-b[i])**2
        //  distance = distance** (1/2)
        //  return distance

        //        from ctypes import pointer
        //def distances(point, a):
        //  dist_list = np.linspace(0,0,len(a))
        //  for i in range(len(a)) :
        //    dist_list[i] = eucledian_distance(p, a[i])
        //  return dist_list

        //    def neighbors_select(distance_list, k):
        //distance = np.sort(distance_list)
        //return distance[:k]

        //        from sklearn.naive_bayes import GaussianNB
        //from sklearn.datasets import make_blobs
        //X, y = make_blobs(n_samples = 100, n_features = 2, centers = 2)
        //g = GaussianNB()
        //g.fit(X_train, y_train)
        //y_pred = g.predict(X_train)

        //from sklearn.cluster import KMeans
        //        k_means = KMeans(n_clusters = 2)
        //k_means.fit(X)

        //labels = k_means.labels_
        //centers = k_means.cluster_centers_
        //            plt.scatter(X[:, 0], X[:, 1], c= labels)
        //plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.5)
        //plt.show()


        //        k_range = range(1, 10)
        //wcss = []
        //for k in k_range:
        //    kmeans = KMeans(n_clusters= k)
        //    kmeans.fit(X)
        //    wcss.append(kmeans.inertia_)
        //    labels = kmeans.labels_
        //    centers = kmeans.cluster_centers_
        //    plt.scatter(X[:, 0], X[:, 1], c=labels)
        //    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.5)
        //    plt.show()
        //plt.plot(k_range, wcss)
        //plt.xlabel('Number of clusters')
        //plt.ylabel('WCSS')
        //plt.show()



        //        import numpy as np
        //def sigmoid(x) :
        //  return 1/(1+np.exp(-x))
        //def sigmoid_derivative(x):
        //  return x* (1-x)

  //      class NeuralNetworks :

  //def init(self, in_size, hid_size, out_size):
  //  self.weights_input_hidden = np.random.rand(in_size, hid_size)
  //  self.weights_output_hidden = np.random.rand(hid_size, out_size)
  //  self.bias_hidden = np.zeros((1, hid_size))
  //  self.bias_output = np.zeros((1, out_size))

  //def forward(self, inputs):
  //  self.hidden_layer_input = np.dots(inputs, self.weights_input_hidden) + self.bias_hidden
  //  self.hidden_layer_output = sigmoid(self.hidden_layer_input)
  //  self.output_layer_input = np.dots(self.hidden_layer_output, self.weights_output_hidden) + self.bias_output
  //  self.output_layer_output = sigmoid(self.output_layer_input)
  //  return self.output_layer_output

  //      def backward(self, outputs, expected):
  //  error_output = expected - outputs
  //  error_hidden_layer = np.dot(error_output, self.weights_output_hidden.T)
  //  delta_output = error_output * sigmoid_derivative(self.output_layer_output)
  //  delta_hidden_layer = error_hidden_layer * sigmoid_derivative(self.hidden_layer_output)
  //  update_input_hidden = np.dot(self.hidden_layer_output.T, delta_hidden_layer)
  //  update_output_hidden = np.dot(self.output_layer_output.T, delta_output)
  //  self.weights_input_hidden += update_input_hidden
  //  self.weights_output_hidden += update_output_hidden
  //  self.bias_hidden += np.sum(delta_hidden_layer, axis= 0, keepdims= True)
  //  self.bias_output += np.sum(delta_output, axis=0, keepdims=True)

    }
}
