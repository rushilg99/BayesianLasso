import numpy as np 
from sklearn.linear_model import Lasso,LassoCV
import statsmodels.api as sm 
import scipy.stats as stats

class BayesianLasso:

	def __init__(self,X_train,y_train):
		self.X = X_train
		self.y = y_train
		self.n = X_train.shape[0]
		self.p = X_train.shape[1]

	def __betaUpdate(self,oldTau2,oldSig2):
		Dtau = np.diag(oldTau2)
		Dtauinv = np.linalg.inv(Dtau)
		A = np.matmul(self.X.T,self.X) + Dtauinv
		Ainv = np.linalg.inv(A)
		normVar = oldSig2*Ainv
		normMean = np.reshape(np.matmul(np.matmul(Ainv,self.X.T),self.y-np.mean(self.y)),(-1,))
		return np.reshape(stats.multivariate_normal.rvs(mean=normMean,cov=normVar),(1,-1))

	def __sig2Update(self,newBeta,oldTau2):
		beta = np.reshape(newBeta,(-1,1))
		Dtauinv = np.linalg.inv(np.diag(oldTau2))
		y_twiddle = self.y - np.mean(self.y)
		
		postshape = 0.5*(self.n-1) + 0.5*(self.p)
		RSS = np.matmul((y_twiddle-np.matmul(self.X,beta)).T,y_twiddle-np.matmul(self.X,beta))
		postscale = 0.5*RSS+0.5*np.matmul(beta.T,np.matmul(Dtauinv,beta))
		return stats.invgamma.rvs(a=postshape,loc=0,scale=postscale)

	def __tau2Update(self,oldLam,newBeta,newSig2):
		newTau2 = np.zeros(len(newBeta))
		for j in range(len(newBeta)):
			mean_j = np.sqrt((oldLam**2)*newSig2/(newBeta[j]**2))
			newTau2[j] = 1/np.random.wald(mean_j,oldLam**2)
		return np.reshape(newTau2,(1,-1))

	def __lambdaUpdate(self,newTau2,shape,delta):
		rate = delta + 0.5*np.sum(newTau2)
		return np.sqrt(stats.gamma.rvs(shape,scale = 1/rate))

	def gibbs_fixedLam(self,lam,n_iter=10000):
		# Create Trace Storage
		beta_traces = np.zeros((n_iter,self.p))
		tau2_traces = np.zeros((n_iter,self.p))
		sig2_traces = np.zeros(n_iter)
		# Set Initial Values
		beta_traces[0,:] = np.ones(self.p)
		tau2_traces[0,:] = np.ones(self.p)
		sig2_traces[0] = 1
		# Start the Gibbs Sampler
		for i in range(n_iter-1):
			beta_traces[i+1,:] = self.__betaUpdate(tau2_traces[i,:],sig2_traces[i])
			sig2_traces[i+1] = self.__sig2Update(beta_traces[i+1,:],tau2_traces[i,:])
			tau2_traces[i+1,:] = self.__tau2Update(lam,beta_traces[i+1,:],sig2_traces[i+1])
		return beta_traces, sig2_traces, tau2_traces

	def gibbs_gammaHPrior(self,r=0.01,delta=0.01,n_iter=10000):
		# Create Trace Storage
		beta_traces = np.zeros((n_iter,self.p))
		tau2_traces = np.zeros((n_iter,self.p))
		sig2_traces = np.zeros(n_iter)
		lambda_traces = np.zeros(n_iter)
		# Set Initial Values
		beta_traces[0,:] = np.ones(self.p)
		tau2_traces[0,:] = np.ones(self.p)
		sig2_traces[0] = 1
		lambda_traces[0] = 1
		# Start the Gibbs Sampler
		for i in range(n_iter-1):
			beta_traces[i+1,:] = self.__betaUpdate(tau2_traces[i,:],sig2_traces[i])
			sig2_traces[i+1] = self.__sig2Update(beta_traces[i+1,:],tau2_traces[i,:])
			tau2_traces[i+1,:] = self.__tau2Update(lambda_traces[i], beta_traces[i+1,:],sig2_traces[i+1])
			lambda_traces[i+1] = self.__lambdaUpdate(tau2_traces[i+1,:],self.p + r, delta)

		return beta_traces, sig2_traces, tau2_traces, lambda_traces 

	def lambda_MML(self,gibbs_iter=10000,em_iter=50,burnin=2000):
		# Create Linear Regression Model
		# Calculate initial lambda for the EM algorithm
		if self.n >= self.p:
			linearModel = sm.OLS(self.y-np.mean(self.y),self.X).fit()
			l1modelparams = np.linalg.norm(linearModel.params,1)
			s2 = np.sum(linearModel.resid**2)/(self.n-self.p)
			lambdainit = self.p*np.sqrt(s2)/l1modelparams
		else:
			crossValidLasso = LassoCV(cv=10,fit_intercept=False,max_iter=10000,alphas=[(0.025*i)/(2*self.n) for i in range(1,4001)]).fit(self.X,np.ravel(self.y-np.mean(self.y)))
			betaLassoParams = np.reshape(crossValidLasso.coef_,(-1,1))
			RSS = np.sum((self.y-np.matmul(self.X,betaLassoParams))**2)
			s2 = RSS/(self.n-np.count_nonzero(np.ravel(betaLassoParams)))
			print("CV Lambda:", crossValidLasso.alpha_*2*self.n)
			print("Smallest Lambda Tried:", np.min(crossValidLasso.alphas_)*2*self.n)
			print("Largest Lambda Tried:", np.max(crossValidLasso.alphas_)*2*self.n)
			print("Beta:", betaLassoParams)
			lambdainit = self.p*np.sqrt(s2)/np.linalg.norm(betaLassoParams,1)
			
		# Start EM algorithm
		lambdas = [lambdainit] 
		for i in range(em_iter):
			betas, sig2s, tau2s = self.gibbs_fixedLam(lambdas[i],gibbs_iter)
			burnedBetas, burnedSig2s, burnedTau2s = betas[burnin:,:], sig2s[burnin:], tau2s[burnin:,:]
			expectedTau2s = np.mean(burnedTau2s,axis=0)
			lambdas.append(np.sqrt(2*self.p/(np.sum(expectedTau2s))))

		# Output path of lambdas from the algorithm, and the MML estimate
		return lambdas,lambdas[-1]