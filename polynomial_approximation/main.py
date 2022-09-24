#%%
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#%%

class Approximator():
    
    #activ has at least 2 elements, inner and outer, or the same number as
    # depth + 1, representing activations at each layer
    
    # network size is (width, depth)
    
    # xlim are bounds for where data is generated from for self.f
    def __init__(self, dim=(1,1), network_size = (10, 1), xlim=(-10,10),
                 activ=["sigmoid"]):
        # dim of domain, range approximated function
        self.dim = dim
        
        # width, depth
        self.network_size = network_size
        self.width = network_size[0]
        self.depth = network_size[1]
        
        # should be called later
        self.model = None
        
        self.optimizer = optimizer=tfk.optimizers.RMSprop(learning_rate=0.01)
        self.xlim=xlim
        
        if len(activ) == 1:
            print("using elt in activ as inner activation")
        elif len(activ) == self.depth:
            print("reading activ as by-layer activations")
        else:
            print("activ passed does not work")
            
        self.activ = activ

        
    def compileModel(self):
        model = tfk.models.Sequential()
        
        model.add(Dense(self.width, 
                        activation=self.activ[0], 
                        input_shape=(self.dim[0],),
                        kernel_initializer=
                        tfk.initializers.RandomNormal(mean=0, stddev = 5)
                        )
                  )
        
        for j in range(self.depth-1):
            if len(self.activ) == self.depth:
                model.add(Dense(self.width, 
                                activation=self.activ[j+1])
                          )
            else:
                model.add(Dense(self.width, 
                                activation=self.activ[0])
                          )
        
        model.add(Dense(self.dim[1]))
        
        
        model.compile(loss="mse", optimizer=self.optimizer, metrics=["accuracy"])
        
        self.model = model
    
    def train(self, size=100000, epochs=100, batch_size=50):
                
        val =  np.sort(np.random.uniform(self.xlim[0], self.xlim[1], size))
        X_test, Y_test = val, self.f(val)
        
        data = np.sort(np.random.uniform(self.xlim[0], self.xlim[1], size))
        X, Y = data, self.f(data)
        
        history = self.model.fit(X, Y, epochs=epochs, batch_size=batch_size,
                validation_data=(X_test, Y_test))
        
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)

        plt.plot(epochs, loss, 'r', label='training loss')
        plt.plot(epochs, val_loss, 'b', label='validation loss')
        plt.title('model loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
        
    def approximate(self, data):
        return self.model.predict(data)
                
    def setOptimizer(self, opt, lr):
        self.optimizer=opt(learning_rate=lr)
        
    def setFunction(self, f):
        self.f = np.vectorize(f)
    


#%%

def func(x: float):
    return -np.power(x, 3) + 4*x 


if __name__ == "__main__":
    
    appr = Approximator(network_size = (3, 1), 
                        activ=["sigmoid"]
                        )
    
    
    appr.setFunction(func)
    appr.setOptimizer(tfk.optimizers.RMSprop, 0.01)
    appr.compileModel()
    xlim = (-2, 2)
    appr.xlim = xlim
    
    print(appr.f(1))
    
    appr.train(epochs=100, batch_size = 2000, size=int(1e5))
    


#%%
    test = np.sort(np.random.uniform(xlim[0], xlim[1], 100))
    
    plt.plot(test, appr.approximate(test), 'b', linewidth=3.0)
    plt.plot(test, appr.f(test), 'r', linewidth=3.0)

    plt.show()

#%%

    appr.model.summary()
    
    print(appr.model.weights)
    
    appr.model.save('3_neuron_cubic')