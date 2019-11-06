import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm



class KNN(object):


    def __init__(self,Train_X,Test_X,Train_Y,Test_Y,Num_Neighbors=20,metric='Euclidean'):
        """
        :param Train_X: Training Feature X
        :param Test_X:  Testing Feature X
        :param Train_Y: Training Label Y (Not one-hot encoding)
        :param Num_Neighbors: The Number of Nearest Neighbors
        """
        self.Train_X = Train_X
        self.Test_X = Test_X
        self.Train_Y = Train_Y
        self.Test_Y = Test_Y
        self.Num_Neighbors = Num_Neighbors
        self.metric = metric

        tf.reset_default_graph()
        with tf.variable_scope('KNN'):


            self.train=tf.placeholder(tf.float32,[None]+[*self.Train_X.shape][1:])
            self.test=tf.placeholder(tf.float32,[None]+[*self.Test_X.shape][1:])
            self.Label=tf.constant(dtype=tf.int32,value=self.Train_Y)

            # Retrieve distance metric
            self.distance=self.get_distance_metric(self.metric)

            # Find K Nearest Neighbors' Index(Small Batch on training set)
            self.dist_placeholder=tf.placeholder(tf.float32,[None,self.Train_X.shape[0]])
            _,self.KNN=tf.nn.top_k(tf.negative(self.dist_placeholder),k=self.Num_Neighbors)
            self.vote=tf.gather(self.Label,self.KNN)

            # Find K Nearest Neighbors' Index(whole training set)
            _,self.KNN2=tf.nn.top_k(tf.negative(self.distance),k=self.Num_Neighbors)
            self.vote2=tf.gather(self.Label,self.KNN2)


        self.sess = tf.InteractiveSession()

    def get_distance_metric(self,metric_name):
        # Euclidean distance
        if metric_name.lower() == 'euclidean':
            distance = tf.reduce_mean(tf.squared_difference(self.train,tf.expand_dims(self.test,1)),2)

        # Variance Normalized Euclidean distance
        elif metric_name.lower() == 'vne':
            distance = tf.squared_difference(self.train,tf.expand_dims(self.test,1))
            _,var = tf.nn.moments(self.train,axes=0)
            distance = tf.sqrt(tf.reduce_sum(tf.divide(distance,var+1e-9),2))

        # Cosine distance
        elif metric_name.lower() == 'cosine':
            normalize_train = tf.nn.l2_normalize(self.train,axis=1)
            normalize_test = tf.nn.l2_normalize(self.test,axis=1)
            distance = 1-tf.reduce_sum(tf.multiply(normalize_train,tf.expand_dims(normalize_test,1)),2)

        # Correlation distance
        elif metric_name.lower() == 'correlation':
            centered_train = tf.subtract(self.train,tf.reduce_mean(self.train,1,keep_dims=True))
            centered_test = tf.subtract(self.test,tf.reduce_mean(self.test,1,keep_dims=True))
            normalize_train = tf.nn.l2_normalize(centered_train,axis=1)
            normalize_test = tf.nn.l2_normalize(centered_test,axis=1)
            distance = 1-tf.reduce_sum(tf.multiply(normalize_train,tf.expand_dims(normalize_test,1)),2)

        # Manhattan distance
        elif metric_name.lower() == 'manhattan':
            distance = tf.reduce_mean(tf.abs(tf.subtract(
                self.train, tf.expand_dims(self.test, 1))), 2)

        # Canberra distance
        elif metric_name.lower() == 'canberra':
            numerator = tf.abs(tf.subtract(self.train, tf.expand_dims(self.test, 1)))
            denominator = tf.add(tf.abs(self.train),
                                tf.expand_dims(tf.abs(self.test), 1))+1e-9
            distance = tf.reduce_sum(numerator/denominator, 2)

        # Braycurtis distance
        elif metric_name.lower() == 'braycurtis':
            numerator = tf.abs(tf.subtract(self.train, tf.expand_dims(self.test, 1)))
            denominator = tf.add(self.train, tf.expand_dims(self.test, 1))+1e-9
            distance = tf.reduce_sum(numerator/denominator, 2)

        else:
            print('Input Metric is not supported')

        return distance

    @staticmethod
    def minibatcher(X, batch_size, shuffle=False):
        """
        :param X: Input feature X
        :param batch_size: size of the small batch
        :param shuffle:  True: randomly sampled small batch, False: In order
        :return: An iterator with small batch of X
        """
        n_samples = X.shape[0]

        if shuffle:
            # Shuffle row ids
             idx = np.random.permutation(n_samples)
        else:
            idx = list(range(n_samples))
        # Partition row ids into small batches
        for k in range(int(np.ceil(n_samples / batch_size))):
            from_idx = k * batch_size
            to_idx = (k + 1) * batch_size
            yield X[idx[from_idx:to_idx]]

    def fit_pred(self,batch_size_test=50,batch_size_train=None):
        """

        :param batch_size: The size of the small batch
        Compute and find nearest neighbor's Index and ground truth label

        Find calss has the most votes from neighbors' vote
        store prediction result in the pred buffer
        calculate accuracy and save result in accuracy variable
        """
        # buffer to save neighbors' labels
        self.Neighbor_Vote = np.zeros([self.Test_X.shape[0],self.Num_Neighbors],dtype=np.uint32)
        # buffer to save neighbors' indexes
        self.Neighbor_index = np.zeros([self.Test_X.shape[0],self.Num_Neighbors],dtype=np.uint32)
        if batch_size_train == None:
            batch_size_train = self.Train_X.shape[0]

        # Partition test set into small batch
        for i,batch_test_X in tqdm(enumerate(self.minibatcher(self.Test_X,batch_size_test))):
            # use whole training features as input
            if batch_size_train == self.Train_X.shape[0]:
                temp_vote,temp_ind = self.sess.run([self.vote2,self.KNN2],
                                     {self.test: batch_test_X,
                                      self.train: self.Train_X})
            # Use small batch training features as input
            else:
                temp_dist = np.empty([batch_size_test,self.Train_X.shape[0]])
                # Partition training set into small batches
                for j,batch_train_X in enumerate(self.minibatcher(self.Train_X, batch_size_train)):
                    # Find small batch distance , shape should be (batch_size_test,batch_size_train)
                    temp = self.sess.run(self.distance,
                                                {self.test:batch_test_X,
                                                 self.train:batch_train_X})

                    temp_dist[:,j*batch_size_train:(j+1)*batch_size_train] = temp

                # Find K Nearest Neighbors' index and Neighbors' Label
                temp_vote,temp_ind=self.sess.run([self.vote,self.KNN],{self.dist_placeholder: temp_dist})

            self.Neighbor_index[i*batch_size_test:(i+1)*batch_size_test] = temp_ind
            self.Neighbor_Vote[i*batch_size_test:(i+1)*batch_size_test] = temp_vote


        # buffer to save prediction result
        self.pred=np.zeros(self.Test_X.shape[0],dtype=np.uint32)

        for i,row in enumerate(self.Neighbor_Vote):
            indice, count = np.unique(row, return_counts=True)
            temp = indice[np.argmax(count)]
            self.pred[i] = temp


        self.accuracy= (self.pred==self.Test_Y.flatten()).mean()




if __name__ =='__main__':

    df=pd.read_csv('C:\\Users\\DataFrame\\Desktop\\SingleClass(GPU)\\SourceFiles\\MSEUMAP_3.csv')
    X=df.filter(regex='UMAP').values
    Y=df.GT_Label.values


    knn=KNN(X,X,Y,Y,Num_Neighbors=20,metric='euclidean')
    knn.fit_pred(batch_size_test=1000)
    print(knn.accuracy)
    # knn.Neighbor_index
    # knn.Neighbor_Vote








