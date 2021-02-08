import pandas as pd
import numpy as np
import numpy.matlib
import faiss

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._base import _set_random_states
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_random_state
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import all_estimators


class FaissKNNClassifier:
    """ Scikit-learn wrapper interface for Faiss KNN.
    Parameters
    ----------
    n_neighbors : int (Default = 5)
                Number of neighbors used in the nearest neighbor search.
    n_jobs : int (Default = None)
             The number of jobs to run in parallel for both fit and predict.
              If -1, then the number of jobs is set to the number of cores.
    algorithm : {'brute', 'voronoi'} (Default = 'brute')
        Algorithm used to compute the nearest neighbors:
            - 'brute' will use the :class: `IndexFlatL2` class from faiss.
            - 'voronoi' will use :class:`IndexIVFFlat` class from faiss.
            - 'hierarchical' will use :class:`IndexHNSWFlat` class from faiss.
        Note that selecting 'voronoi' the system takes more time during
        training, however it can significantly improve the search time
        on inference. 'hierarchical' produce very fast and accurate indexes,
        however it has a higher memory requirement. It's recommended when
        you have a lots of RAM or the dataset is small.
        For more information see: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    n_cells : int (Default = 100)
        Number of voronoi cells. Only used when algorithm=='voronoi'.
    n_probes : int (Default = 1)
        Number of cells that are visited to perform the search. Note that the
        search time roughly increases linearly with the number of probes.
        Only used when algorithm=='voronoi'.
    References
    ----------
    Johnson Jeff, Matthijs Douze, and Hervé Jégou. "Billion-scale similarity
    search with gpus." arXiv preprint arXiv:1702.08734 (2017).
    """

    def __init__(self,
                 n_neighbors=5,
                 n_jobs=None,
                 algorithm='brute',
                 n_cells=100,
                 n_probes=1):

        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        self.n_cells = n_cells
        self.n_probes = n_probes

        import faiss
        self.faiss = faiss

    def predict(self, X):
        """Predict the class label for each sample in X.
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        preds : array, shape (n_samples,)
                Class labels for samples in X.
        """
        idx = self.kneighbors(X, self.n_neighbors, return_distance=False)
        class_idx = self.y_[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes_), axis=1,
            arr=class_idx.astype(np.int16))
        preds = np.argmax(counts, axis=1)
        return preds

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.
        n_neighbors : int
            Number of neighbors to get (default is the value passed to the
            constructor).
        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned
        Returns
        -------
        dists : list of shape = [n_samples, k]
            The distances between the query and each sample in the region of
            competence. The vector is ordered in an ascending fashion.
        idx : list of shape = [n_samples, k]
            Indices of the instances belonging to the region of competence of
            the given query sample.
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        elif n_neighbors <= 0:
            raise ValueError("Expected n_neighbors > 0."
                             " Got {}" .format(n_neighbors))
        else:
            if not np.issubdtype(type(n_neighbors), np.integer):
                raise TypeError(
                    "n_neighbors does not take {} value, "
                    "enter integer value" .format(type(n_neighbors)))

        check_is_fitted(self, 'index_')

        X = np.atleast_2d(X).astype(np.float32)
        dist, idx = self.index_.search(numpy.ascontiguousarray(X), n_neighbors)
        if return_distance:
            return dist, idx
        else:
            return idx

    def predict_proba(self, X):
        """Estimates the posterior probabilities for sample in X.
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        preds_proba : array of shape (n_samples, n_classes)
                          Probabilities estimates for each sample in X.
        """
        idx = self.kneighbors(X, self.n_neighbors, return_distance=False)
        class_idx = self.y_[idx]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.n_classes_), axis=1,
            arr=class_idx.astype(np.int16))

        preds_proba = counts / self.n_neighbors

        return preds_proba

    def fit(self, X, y):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Data used to fit the model.
        y : array of shape (n_samples)
            class labels of each example in X.
        """
        X = np.atleast_2d(X).astype(np.float32)
        X = np.ascontiguousarray(X)
        d = X.shape[1]  # dimensionality of the feature vector
        self._prepare_knn_algorithm(X, d)
        self.index_.add(X)
        self.y_ = y
        self.n_classes_ = np.unique(y).size
        return self

    def _prepare_knn_algorithm(self, X, d):
        if self.algorithm == 'brute':
            self.index_ = self.faiss.IndexFlatL2(d)
        elif self.algorithm == 'voronoi':
            quantizer = self.faiss.IndexFlatL2(d)
            self.index_ = self.faiss.IndexIVFFlat(quantizer, d, self.n_cells)
            self.index_.train(X)
            self.index_.nprobe = self.n_probes
        elif self.algorithm == 'hierarchical':
            self.index_ = self.faiss.IndexHNSWFlat(d, 32)
            self.index_.hnsw.efConstruction = 40
        else:
            raise ValueError("Invalid algorithm option."
                             " Expected ['brute', 'voronoi', 'hierarchical'], "
                             "got {}" .format(self.algorithm))
    
    def faiss_kdn_score(self, X, y, k):

        nbrs = FaissKNNClassifier(n_neighbors=k + 1, algorithm='brute', n_jobs=-1).fit(X,y)
        _, indices = nbrs.kneighbors(X)
        neighbors = indices[:, 1:]
        diff_class = np.matlib.repmat(y, k, 1).transpose() != y[neighbors]
        score = np.sum(diff_class, axis=1) / k
        return score
    


class InstanceHardness():
    
  

    def ensemble_hardness(self, X, y, estimator, random_state=42, cv=5):
            
        random_state = check_random_state(random_state)

        skf = StratifiedKFold(
            n_splits= cv, shuffle=True, random_state=random_state,
        )
        probabilities = cross_val_predict(
            estimator, X, y, cv=skf, n_jobs=-1,
            method='predict_proba'
        )
        probabilities = probabilities[range(len(y)), y]

        hardness = np.subtract(1, probabilities)
        
        curriculum_df = pd.DataFrame(hardness, columns = ['score'])
        curriculum_df.reset_index(inplace=True)
        
        return curriculum_df



    def GMM_IH(self, X, y):


        #Define the number of Gaussians to be the same as classes
        n_classes = len(np.unique(y))

        #Fit GMM and generate prediction probabilities for each example
        gmm = GaussianMixture(n_components=n_classes, covariance_type='tied')
        gmm.fit(X)
        scores = gmm.predict_proba(X)
        ih = 1 - scores

        #For each example select the lowest probability score out of all Gaussians to be it's IH score.
        #Higher IH scores indicate examples in overlapping gaussian areas or outliers.
        instance_hardness = []
        for i in range(0, len(ih)):
            instance_hardness.append(min(ih[i]))

        #Generate dataframe with scores and indices for every example.
        curriculum_df = pd.DataFrame(instance_hardness, columns = ['score'])
        curriculum_df.reset_index(inplace=True)

        return curriculum_df


    def kdn_score(self, X, y, k):
        """
        Calculates the K-Disagreeing Neighbors score (KDN) of each sample in the
        input dataset.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.

        y : array of shape (n_samples)
            class labels of each example in X.

        k : int
            Neighborhood size for calculating the KDN score.

        Returns
        -------

        score : array of shape = [n_samples,1]
            KDN score of each sample in X.

        neighbors : array of shape = [n_samples,k]
            Indexes of the k neighbors of each sample in X.


        References
        ----------
        M. R. Smith, T. Martinez, C. Giraud-Carrier, An instance level analysis of
        data complexity,
        Machine Learning 95 (2) (2014) 225-256.

        """

        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree', n_jobs=-1).fit(X)
        _, indices = nbrs.kneighbors(X)
        neighbors = indices[:, 1:]
        diff_class = np.matlib.repmat(y, k, 1).transpose() != y[neighbors]
        score = np.sum(diff_class, axis=1) / k
        return score

    def switch_curriculum(self):
        switch = True
        return switch

    def anti_curriculum(self):
        switch = False
        return switch

    def load_curriculum(self):
	#Nota pra mim: Para o KDN usar FAISS!
        curriculum_df = pd.read_csv('/home/administrator/nil/notebooks/datasets/class/curriculo_jasmine_gmm.csv')
        return curriculum_df


    def generate_curriculum(self, X, y, path, method):


        X = np.array(X)
        y = np.array(y)
        
        if method == 'kdn':
            score = self.kdn_score(X, y, 50)
            curriculum_df = pd.DataFrame(score, columns = ['score'])
            curriculum_df.reset_index(inplace=True)
            curriculum_df.to_csv(path, index=False)
        elif method == 'faiss_kdn':
            score = FaissKNNClassifier().faiss_kdn_score(X, y, 50)
            curriculum_df = pd.DataFrame(score, columns = ['score'])
            curriculum_df.reset_index(inplace=True)
            curriculum_df.to_csv(path, index=False)
        elif method == 'gmm':
            curriculum_df = self.GMM_IH(X, y)
            curriculum_df.to_csv(path, index=False)
            
        elif method == 'ensemble':
            estimators = all_estimators(type_filter='classifier')

            clf_l = ["RandomForestClassifier", "MLPClassifier", "SVC"]
 
            classifiers = []
            for name, class_ in estimators:
                if hasattr(class_, 'predict_proba') and name in clf_l:
                    if name == "SVC":
                        clf = class_(probability=True)
                    else:
                        clf = class_()
                    classifiers.append(clf)
               
            estimator = VotingClassifier(estimators=[('mlp',classifiers[0]),('rf', classifiers[1]), ('svm', classifiers[2])],
                                     voting='soft')
            curriculum_df = self.ensemble_hardness(X, y, estimator)
            curriculum_df.to_csv(path, index=False)
        else:
            print("Aborting generation")
            

        return


