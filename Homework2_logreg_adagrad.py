'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''
import numpy as np 
import random
import math

class LogisticRegressionAdagrad:

    def __init__(self, alpha = 0.01, regLambda=0.01, regNorm=2, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        Arguments:
        	alpha is the learning rate
        	regLambda is the regularization parameter
        	regNorm is the type of regularization (either L1 or L2, denoted by a 1 or a 2)
        	epsilon is the convergence parameter
        	maxNumIters is the maximum number of iterations to run
        '''
        self.alpha=alpha
        self.regLambda=regLambda
        self.regNorm=regNorm
        self.epsilon=epsilon
        self.maxNumIters=maxNumIters
        self.theta=None

    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        n,d=X.shape
        Sig_out=self.sigmoid(X*theta)
        #regular=regLambda * np.ones((d,1))
        #regular[0,0]=0
        if self.regNorm == 2:
            regular_theta=np.linalg.norm(theta,2)
        elif self.regNorm == 1  :
            regular_theta=np.linalg.norm(theta,1)           
#        J =  -((y).T*np.log(Sig_out)+(1-y).T * np.log(1-Sig_out)) + 0.5*regular.T*regular_theta
        J = (-y.T * np.log(self.sigmoid(X * theta)) - (1.0 - y).T * 
            np.log(1.0 - self.sigmoid(X * theta))) + 0.5*regLambda * (regular_theta)
        J_scalar = J.tolist()[0][0]  # convert matrix to scalar
        #print(J_scalar)
        return J_scalar
    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
       
       
        n,d = X.shape
#        regular=regLambda * np.ones((d,1))
#        regular[0,0]=0
        #gradient = (X.T*((self.sigmoid(np.matmul(X,theta)))-y)) +(regular.T*theta)
        if self.regNorm ==1:
            gradient = (X.T * (self.sigmoid(X * theta) - y) + regLambda*np.sign(theta))
        else:
            gradient = (X.T * (self.sigmoid(X * theta) - y) + regLambda*theta)
        # don't regularize the theta_0 parameter
        gradient[0] = sum(self.sigmoid(X * theta) - y)
        
        return gradient
    

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
       # self.regLambda=0.0001
        n =X.shape[0]
        X = np.c_[np.ones((n,1)), X]
        n,d = X.shape
        Grad_hist =np.zeros((d,1))
#        theta_new=np.zeros((d,1))
        self.theta = np.matrix(np.random.randn((d))).T
        for i in range(self.maxNumIters):
            m=np.random.randint(0,n)
            X_new=X[m,:].reshape(1,d)
            y_new=y[m].reshape(1,1)
            grad = self.computeGradient(self.theta, X_new, y_new, self.regLambda)
            Grad_hist=Grad_hist+np.power(grad,2)
            for k in range(0,d):
                learn_rate=self.alpha/(math.sqrt(Grad_hist[k])+self.epsilon)
                theta_new=self.theta-(learn_rate * grad)
                #cost=self.computeCost(self.theta,X, y,self.regLambda) 
                self.theta=theta_new
#            theta_new = self.theta - self.alpha*self.computeGradient(self.theta, X_new, y_new, self.regLambda)
#            self.theta=theta_new
            ########## adagarad
            

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        n,d=X.shape
        X = np.c_[np.ones((n,1)), X]
        y_final=self.sigmoid(np.matmul(X,self.theta))
        for i in range(y_final.shape[0]):
            if y_final[i] > 0.5:
                y_final[i]=1
            elif y_final[i]<= 0.5:
                y_final[i]=0
        return np.array(y_final)


    def sigmoid(self, Z):
     
        return 1/(1+np.exp(-Z))
