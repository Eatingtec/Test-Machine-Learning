import numpy as np
from sklearn.preprocessing import Imputer

class UCIDataset(object):
    def __inti__(self,data=None,target=None):
        self.data = data
        self.target = target



class UCIDatasetsLoader(object):
    def __init__(self):
        pass
    
    def load_abalone(self):
        dataset = UCIDataset()
        martrix = np.loadtxt(fname="UCI datasets/abalone.data",delimiter=",")
        dataset.data = np.array(martrix[:,0:8],dtype='float64')
        dataset.target = np.array(martrix[:,8],dtype='int32')
        return dataset
    
    #with missing data
    def load_breast_cancer_wisconsin(self):
        dataset = UCIDataset()
        martrix = np.loadtxt(fname="UCI datasets/breast-cancer-wisconsin.data",delimiter=",")
        dataset.data = np.array(martrix[:,1:10],dtype='float64')
        dataset.target = np.array(martrix[:,10],dtype='int32')
        imputer = Imputer(strategy='most_frequent')
        dataset.data = imputer.fit_transform(dataset.data)
        dataset.data = dataset.data.astype('int32')
        return dataset

    def load_glass(self):
        dataset = UCIDataset()
        martrix = np.loadtxt(fname="UCI datasets/glass.data",delimiter=",")
        dataset.data = np.array(martrix[:,1:10],dtype='float64')
        dataset.target = np.array(martrix[:,10],dtype='int32')
        return dataset
    
    #with missing data
    def load_hepatitis(self):
        dataset = UCIDataset()
        martrix = np.loadtxt(fname="UCI datasets/hepatitis.data",delimiter=",")
        dataset.data = np.array(martrix[:,1:20],dtype='float64')
        dataset.target = np.array(martrix[:,0:1],dtype='int32')
        dataset.target = dataset.target[:,0]
        imputer = Imputer(strategy='most_frequent')
        dataset.data = imputer.fit_transform(dataset.data)
        dataset.data = dataset.data.astype('int32')
        return dataset
    
    def load_ecoli(self):
        dataset = UCIDataset()
        martrix = np.loadtxt(fname="UCI datasets/ecoli.data",delimiter=",")
        dataset.data = np.array(martrix[:,0:7],dtype='float64')
        dataset.target = np.array(martrix[:,7],dtype='int32')
        return dataset
    
    def load_iris(self):
        dataset = UCIDataset()
        martrix = np.loadtxt(fname="UCI datasets/iris.data",delimiter=",")
        dataset.data = np.array(martrix[:,0:4],dtype='float64')
        dataset.target = np.array(martrix[:,4],dtype='int32')
        return dataset
    
    #with missing data
    def load_lung_cancer(self):
        dataset = UCIDataset()
        martrix = np.loadtxt(fname="UCI datasets/lung-cancer.data",delimiter=",")
        dataset.data = np.array(martrix[:,1:57],dtype='float64')
        dataset.target = np.array(martrix[:,0:1],dtype='int32')
        dataset.target = dataset.target[:,0]
        imputer = Imputer(strategy='most_frequent')
        dataset.data = imputer.fit_transform(dataset.data)
        dataset.data = dataset.data.astype('int32')
        return dataset
    
    def load_transfusion(self):
        dataset = UCIDataset()
        martrix = np.loadtxt(fname="UCI datasets/transfusion.data",delimiter=",",skiprows=1)
        dataset.data = np.array(martrix[:,0:4],dtype='int32')
        dataset.target = np.array(martrix[:,4],dtype='int32')
        return dataset
    
    def load_winequality_red(self):
        dataset = UCIDataset()
        martrix = np.loadtxt(fname="UCI datasets/winequality-red.csv",delimiter=";",skiprows=1)
        dataset.data = np.array(martrix[:,0:11],dtype='float64')
        dataset.target = np.array(martrix[:,11],dtype='int32')
        return dataset
    
    def load_winequality_white(self):
        dataset = UCIDataset()
        martrix = np.loadtxt(fname="UCI datasets/winequality-white.csv",delimiter=";",skiprows=1)
        dataset.data = np.array(martrix[:,0:11],dtype='float64')
        dataset.target = np.array(martrix[:,11],dtype='int32')
        return dataset
    


datasets_loader = UCIDatasetsLoader()