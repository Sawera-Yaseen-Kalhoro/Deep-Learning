from data import *
from sklearn.svm import SVC

class KSVWrap(SVC):
  def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
    """
        Constructs the wrapper and trains the RBF SVM classifier
    
        X,Y_:            data and indices of correct data classes
        param_svm_c:     relative contribution of the data cost
        param_svm_gamma:  RBF kernel width
    """
    super(KSVWrap, self).__init__()
    self.X = X
    self.Y_ = Y_
    self.param_svm_gamma = param_svm_gamma
    self.param_svm_c = param_svm_c
    self.model = SVC(C = self.param_svm_c, gamma = param_svm_gamma, kernel = 'rbf')
    self.model.fit(self.X, self.Y_)
   

  def predict(self, X):
    #Predicts and returns the class indices of data X
    return self.model.predict(X)


  def get_scores(self, X):
    # Returns the classification scores of the data
    return self.model.decision_function(X)


  def suport (self):
    # Indices of data chosen as support vectors
    return self.model.support_


def eval(model, X):
   return model.get_scores(X)


def decfun(model):
    def classify(X):
       return eval(model, X)
    return classify


if __name__ == "__main__":
   np.random.seed(100)
   X, Y_ = sample_gmm_2d(6,2,10)
   model = KSVWrap(X, Y_, param_svm_gamma = 'auto', param_svm_c = 1)
   Y = model.predict(X)

   average_precision = eval_AP(Y)
   print("Average precision: {}".format(average_precision)) 
   accuracy, pr, M = eval_perf_multi(Y, Y_)
   print("Accuracy:", accuracy)
   for i, (recall, precision) in enumerate(pr):
         print("Class {}: Recall = {:.2f}, Precision = {:.2f}".format(i, recall, precision))

   decfun = decfun(model)
   bbox = (np.min(X, axis = 0), np.max(X, axis = 0))
   graph_surface(decfun, bbox, offset = 0)
   graph_data(X, Y_, Y, special = model.suport())
   plt.show()
  
