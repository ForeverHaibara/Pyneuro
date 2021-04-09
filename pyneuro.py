import activations
import numpy as np
import pandas as pd

class Neuro():
    def __init__(self,cellnum,learning_rate=0.1,activators=1):
        self.cellnum = cellnum
        self.layer_num = len(cellnum)-1
        self.learning_rate = learning_rate
        
        if type(activators) == int :
            self.activator = [activations.activator(activators)] * self.layer_num
        else :
            self.activator = [activations.activator(activator) for activator in activators]
        
        self.w , self.b = [] , [] # weights and biases
        for height, width in zip(cellnum[:-1],cellnum[1:]):
            self.w.append(np.random.normal(0,1,(height,width)))
            self.b.append(np.random.normal(0,1,(1,width)))


    def fit(self,_x,_y,epochs,batch_size,learning_rate=None,loss_function=0):
        if learning_rate:
            self.learning_rate = learning_rate
        # credit_learning_rate = self.learning_rate / batch_size
        w_update = [ np.zeros(weight.shape) for weight in self.w ]
        b_update = [ np.zeros(bias.shape) for bias in self.b ]

        for epoch in range(epochs):
            self.learning_rate *= 0.9   # learning-rate decay
            index = np.random.permutation(_x.shape[0])

            # activated value, x[layer].shape = ( batch_size , cell[layer] )
            x = [ np.empty((batch_size,width)) for width in self.cellnum ]

            # pre-activated value , z[layer].shape = ( batch_size , cell[layer+1])
            z = [ np.empty((batch_size,width)) for width in self.cellnum[1:] ]

            # error[layer].shape = ( batch_size , cell[layer+1] )
            error = [ np.empty((batch_size,width)) for width in self.cellnum[1:] ]

            for batch_num in range(_x.shape[0],batch_size-1,-batch_size):
                x[0] = _x[index[batch_num-batch_size : batch_num] , :]
                y    = _y[index[batch_num-batch_size : batch_num] , :]

                # forward_propagation
                for layer in range(self.layer_num):
                    # b[layer] should be broadcast to shape (batch_size,b[layer].width()) 
                    # (automatically broadcast by numpy)
                    z[layer] = np.dot(x[layer], self.w[layer]) + self.b[layer]
                    x[layer+1] = self.activator[layer].function()(z[layer])
                
               
                if loss_function == 0 :
                    # l2 loss
                    # hadmard multiply
                    error[-1] = (x[-1] - y) * self.activator[-1].derivative()(z[-1]) * (1.0/batch_size)
                    print("loss = ",np.linalg.norm(x[-1]-y)/batch_size)
                else :
                    # cross-entropy loss
                    error[-1] = (x[-1] - y) * (1.0/batch_size)
                    print("loss = ",np.linalg.norm(y*np.log(x[-1]) + (1-y)*np.log(1-x[-1]))/batch_size)



                # back propagation
                for layer in range(self.layer_num-2,-1,-1):
                    error[layer] = self.activator[layer].derivative()(z[layer]) * np.dot(error[layer+1],self.w[layer+1].transpose())

                # update
                for layer in range(self.layer_num-1,-1,-1):
                    # momentum optimizer
                    w_update[layer] = w_update[layer] * 0 + np.dot(x[layer].transpose() , error[layer]) * self.learning_rate
                    b_update[layer] = b_update[layer] * 0 + np.sum(error[layer],axis=0) * self.learning_rate
                    self.w[layer] -= w_update[layer]    
                    self.b[layer] -= b_update[layer]


    def predict(self,_x):
        for layer in range(self.layer_num):
            # b[layer] should be broadcast to shape (batch_size,b[layer].width()) 
            # (automatically broadcast by numpy)
            z = np.dot(_x, self.w[layer]) + self.b[layer]
            _x = self.activator[layer].function()(z)
        return _x


    def save(self,path):
        #pd.DataFrame({'weight':self.w,'bias':self.b}).to_csv(path,sep='\t' if '.txt' in path else ',',#header = None,#index = None)
        def printData(path,x):
            x = np.array(x)
            with open(path,'a') as f:
                try:
                    f.write(str(x.shape[0])+' '+str(x.shape[1])+' ')
                except:
                    f.write(str(x.shape[0])+' 1 ')
                for i in x.flatten().tolist():
                    f.write(str(i)+' ')
                f.write('\n')
            
        with open(path,'w') as f:
            f.write('')
        
        printData(path,self.cellnum)
        for weights, bias in zip(self.w,self.b):
            printData(path,weights)
            printData(path,bias)
                
            
    def load(self,path):
        def readData(string):
            split_data = [float(i) for i in string.split()]
            return np.reshape(split_data[2:], (round(split_data[0]),round(split_data[1])))

        with open(path,'r') as f:
            data = f.readlines()

        self.cellnum = [ int(cell) for cell in data[0].split()[2:] ]
        self.layer_num = len(self.cellnum) - 1

        self.w , self.b = [] , []
        for layer in range(1,self.layer_num+1):
            self.w.append(readData(data[layer*2-1]))
            self.b.append(readData(data[layer*2]))
        