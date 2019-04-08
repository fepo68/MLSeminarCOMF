import numpy as np # libreria de manejo de datos matriciales y operaciones multivariadas
import math # Libreria de opreaciones matematicas

def linearRegressionModule(X,t,basisFNC,NbF):
    Ndata,D = X.shape
    #print Ndata,D
    yEst = np.zeros((Ndata,1))
    # Calculo de la matriz PHI de funciones base
    PHI = np.zeros((Ndata,NbF+1))
    PHI[:,0] = 1
    for n in range(0,Ndata):
        #print X[n]
        for i in range(1,NbF+1):
            if basisFNC == 'pol':                
                PHI[n][i] = X[n]**(i)
    
    # Luego se estima el mejor W que maximiza la verosimilitud utilizando minimos cuadrados
    PHIT = PHI.T
    w_ml = np.linalg.inv(PHIT.dot(PHI)).dot(PHIT.dot(t))
    yEst = PHI.dot(w_ml)
    #print w_ml
    return PHI,w_ml,yEst
		
	