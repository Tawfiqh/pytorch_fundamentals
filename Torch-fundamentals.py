#!/usr/bin/env python
# coding: utf-8

# In[1]:


#  - create some model parameters for a simple linear model
#      - a torch 3-vector called w
#      - a torch scalar called b
#    - create an example mini-dataset with features called X and labels called y
#      - X should have shape (4, 3) (features, 3 examples)
#      - y should have shape (4)
#      - make up a dummy function for assigning labels e.g. y = sum(features)
#    - create a prediction using the linear regression equation
#    - create a function to compute the MSE loss
#    - compute the derivatives of the loss with respect to all of the variables which contributed to it
#      - get an error? why? remember what kwarg you need to give the tensors which you want to compute the derivatives of when you created them?
#    - should your data (features, labels) have requires_grad=True?
#    - what about your model parameters (w, b)?
#    - try loss.backward() again
#    - check the .grad attribute of the model params
#    - what is the grad_fn of the loss?
#    - does the grad_fn exist for the model params, predictions or features? Discuss why?
#    - run .backward() again, without running the whole script. What error occurs and why?


# In[2]:


import torch
#  - create some model parameters for a simple linear model
#      - a torch 3-vector called w
#      - a torch scalar called b
number_of_features = 3
number_of_labels = 1.0
w = torch.randn(number_of_features, requires_grad=True)  # Random normal
b = torch.tensor([number_of_labels], requires_grad=True)  # create again another random tensor

# y = X @ W + b

print(w)
print(b)


# In[3]:


#    - create an example mini-dataset with features called X and labels called y
#      - X should have shape (4, 3) (features, 3 examples)
number_of_examples = 5
X = torch.rand(number_of_examples, number_of_features)
X


# In[4]:


#    - create an example mini-dataset with features called X and labels called y
#      - y should have shape (4)
#      - make up a dummy function for assigning labels e.g. y = sum(features)

def get_labels(X):
    return torch.sum(X, axis=1)

y = get_labels(X)
y


# In[5]:


#    - create a prediction using the linear regression equation
y_hat = X@w + b
y_hat


# In[6]:



#    - create a function to compute the MSE loss
def mse(y, y_hat):
    return torch.mean((y - y_hat) ** 2)

loss = mse(y,y_hat)
loss_backward = False
loss


# In[7]:


#    - compute the derivatives of the loss with respect to all of the variables
# which contributed to it
print(f"loss_backward:{loss_backward}")
if not loss_backward:
    grad = loss.backward()
    loss_backward = True
print(grad)


# In[8]:


#      - get an error? why? remember what kwarg you need to give the tensors which you want to compute the derivatives of when you created them?
#    - should your data (features, labels) have requires_grad=True? -- Nope - only the bits in between 

#    - what about your model parameters (w, b) - Yes! As they are in the hidden-layers
#    - try loss.backward() again


# In[9]:


print(f"w.grad:{w.grad}")
print(f"b.grad:{b.grad}")
print(f"X.grad:{X.grad}")
print(f"y.grad:{y.grad}")
print(f"y_hat.grad:{y_hat.grad}")


#    - check the .grad attribute of the model params
#    - what is the grad_fn of the loss? -- it is the class/function that calculates the grad for the tensor


#    - does the grad_fn exist for the model params, predictions or features? Discuss why?
#    - run .backward() again, without running the whole script. What error occurs and why?


# In[10]:


class LinearRegressorTorchy(torch.nn.Module):
    def __init__(self, n_features, n_labels):
        super().__init__()
        print(f"n_features:{n_features}")
        print(f"n_labels:{n_labels}")
        self.linear_layer = torch.nn.Linear(n_features, n_labels)
        
    def forward(self, X):
        return self.linear_layer(X)

    
linear_regressor = LinearRegressorTorchy(number_of_features, int(number_of_labels))


# In[20]:


from torch.nn import functional as F

y_hat = linear_regressor(X)
print(f"y_hat:{y_hat}")

loss = F.mse_loss(y, y_hat.reshape(-1))
print(f"loss: {loss}")


# In[40]:


import matplotlib.pyplot as plt

def train(model, X, y, epochs=100, lr=0.1, print_losses=False):
    optimiser = torch.optim.SGD(model.parameters(), lr) # create optimiser
    losses = []
    for epoch in range(epochs):
        optimiser.zero_grad()
        y_hat = model(X)
        loss = F.mse_loss(y_hat.reshape(-1), y.reshape(-1))
        if(print_losses):
            print(f"loss:{loss}")
        loss.backward() # Upgrades the .grad -- of each of the parameters (based on backpopulating through the NN)
        optimiser.step()
        losses.append(loss.item())
    plt.plot(losses)
    plt.show()
    

train(linear_regressor, X, y, epochs=500)


# In[50]:


from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing

X_boston, y_boston = datasets.load_boston(return_X_y=True)

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25) # 0.25 x 0.8 = 0.2


sc = preprocessing.StandardScaler()
sc.fit(X_boston)
X_boston = sc.transform(X_boston)


X_boston = torch.Tensor(X_boston)
y_boston = torch.Tensor(y_boston)

linear_regressor_boston = LinearRegressorTorchy(X_boston.shape[1], 1)
train(linear_regressor_boston, X_boston, y_boston, epochs=500, lr=0.01, print_losses=False)

from sklearn.metrics import r2_score
r2_error = r2_score(linear_regressor_boston(X_boston).detach().numpy(), y_boston.detach().numpy())
print(f"R^2 error: {r2_error}")


# In[28]:


print(y_boston.shape)


# In[51]:


## Move it inside the class:
class LinearRegressorTorchyFit(torch.nn.Module):
    def __init__(self, n_features, n_labels):
        super().__init__()
        print(f"n_features:{n_features}")
        print(f"n_labels:{n_labels}")
        self.linear_layer = torch.nn.Linear(n_features, n_labels)
        
    def forward(self, X):
        return self.linear_layer(X)

    def fit(model, X, y, epochs=100, lr=0.1, print_losses=False):
        optimiser = torch.optim.SGD(model.parameters(), lr) # create optimiser
        losses = []
        for epoch in range(epochs):
            optimiser.zero_grad()
            y_hat = model(X)
            loss = F.mse_loss(y_hat.reshape(-1), y.reshape(-1))
            if(print_losses):
                print(f"loss:{loss}")
            loss.backward() # Upgrades the .grad -- of each of the parameters (based on backpopulating through the NN)
            optimiser.step()
            losses.append(loss.item())
        plt.plot(losses)
        plt.show()

linear_regressor_boston = LinearRegressorTorchyFit(X_boston.shape[1], 1)
linear_regressor_boston.fit(X_boston, y_boston, epochs=500, lr=0.01, print_losses=False)

from sklearn.metrics import r2_score
r2_error = r2_score(linear_regressor_boston(X_boston).detach().numpy(), y_boston.detach().numpy())
print(f"R^2 error: {r2_error}")


# In[ ]:




