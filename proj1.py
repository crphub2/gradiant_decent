import math 
import numpy as np
outliers=[]
def mean(x):
    s=0
    for i in x:
        s=s+i
    return(s/len(x))
    
def SD(a):
    n=len(a)
    var=0
    men=mean(a)
    sq = 0
    for i in range(0 ,n): 
        sq += ((a[i] - men) * (a[i] - men)) 
    var=sq / n 
    return math.sqrt(var)     


def data_Normalization(x):
    mean_1 = mean(x)
    x_nor=[]
    std_1 =SD(x)
    for i in x:
        d_nor= (i - mean_1)/std_1 
        x_nor.append(d_nor)
    return x_nor  


    
def detect_outlier(x):
    threshold=3
    for z_score in x:
        if z_score > threshold or z_score<(-threshold):
            outliers.append(z_score)
    return outliers 
    

def Remove_outlier(x):
    z=detect_outlier(x)
    s=mean(x)
    xx=list(x)
    for i in range(0,len(xx)-1):
        if xx[i] in z:
             xx.pop(i)
             xx.insert(i,s)
    return np.array(xx)


#SETTING HYPERPARAMETER
alpha = 0.01
iters = 1000


# X DENOTE INDEPENDENT VARIABLE SET
# y DENOTE DEPENDENT VARIABLE

#COMPUTING COAST FUNCTION
def computeCost(X,y,theta):
    j_theta = np.power(((np.dot(X,theta.T))-y),2)
    return np.sum(j_theta)/(2 * len(X))



#gradient descent
def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (np.dot(X,theta.T) - y), axis=0)
        cost[i] = computeCost(X, y, theta)
    
    return theta,cost

#running the gd and cost function
g,cost = gradientDescent('INDEPENDENT VARIABLE',theta,iters,alpha)
finalCost = computeCost("component of your data set(INDEPENDENT VARIABLE(X-VALUE/S))",g)
print(finalCost)



# CREATING MODEL
# NOTE - HERE U CAN CREATE MODEL ACCORDING TO COMPLEXITY IT CAN BE LINEAR OR POLYNOMIAL REGRESSION
 
b0=g[0][0] # C VALUE OF EQUATION (BIAS)
b1=g[0][1] # COFFICENT OF INDEPENDENT VALUE
y_mean=mean(y)
sumofsquares = 0
sumofresiduals = 0
for i in range(len(X)) :
    Y_prd=[]
    y_pred = b0 + b1 * X[i]  
    Y_prd.append(y_pred)
    sumofsquares += (y[i] - y_mean) ** 2
    sumofresiduals += (y[i] - y_pred) **2
    
score  = 1 - (sumofresiduals/sumofsquares)
# ABOUT MODEL ACCURACY 
print(score)

# PREDECTED VALUE OF MODEL
Y_prd=np.array(Y_prd) 












