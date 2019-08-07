#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as T
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
#from google.colab import drive                IF you are using COLAB


# In[ ]:


# Use this if you are working on COLAB
# This will prompt for authorization.
#drive.mount('/content/drive')


# In[6]:


def extract_data(x_data_filepath, y_data_filepath):
    X = np.load(x_data_filepath)
    y = np.load(y_data_filepath)
    return X, y
# X,y=extract_data('C:/Users/Rutuja Moharil/hw4_skeleton/HW4/data/images_train.npy','C:/Users/Rutuja Moharil/hw4_skeleton/HW4/data/labels_train.npy')
# data=Dataset(X,y)
# X,y=data.__getitem__(range(len(X)))
# # numimage=len(X)
# # print(X.shape)
# #X_new=X.T


# X_test = np.load('C:/Users/Rutuja Moharil/hw4_skeleton/HW4/data/images_test.npy')
# y_test=np.zeros(len(X_test))
# X_test=torch.from_numpy(X_test).float()
# y_test=torch.from_numpy(y_test).float()
#print(len(X_test))


# In[ ]:



    


# In[1]:


def data_visualization(images,labels):
  
    for i in range(5):
        j=np.array(np.where(labels==i))
        for m in range(6):
            #images=X[:,:,:,j[m]]
         #X=np.swapaxes(X,0,1)
         #print(images[:,:,:,j[0][m]])
            plt.figure()
            plt.imshow(images[:,:,:,j[0][m]])
 
    pass


# In[5]:


############################################################
# Extracting and loading data
############################################################
class Dataset(Dataset):
    def __init__(self, X, y):
        self.len = len(X)           
        if torch.cuda.is_available():
          self.x_data = torch.from_numpy(X).float().cuda()
          self.y_data = torch.from_numpy(y).long().cuda()
        else:
          self.x_data = torch.from_numpy(X).float()
          self.y_data = torch.from_numpy(y).long()
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


# In[15]:


# print(len(X))
# new_x_train,new_y_train,x_val,y_val=create_validation(X,y)
# print(len(new_x_train))


# In[14]:


def create_validation(x_train,y_train):

#    permutation = np.random.permutation(y_train.shape[0])
      #y_train = y_train[permutation]
    #print(permutation)
    indices=np.array(range(len(x_train)))
    permutation=np.random.permutation(indices)

    left_count = int(np.floor((len(x_train)) * 0.8))
    left_count= len(x_train)-left_count
    new_x_train = x_train[permutation[left_count:]]
    new_y_train = y_train[permutation[left_count:]]
    x_val =x_train[permutation[:left_count]]
    y_val=y_train[permutation[:left_count]]
    

    return new_x_train,new_y_train,x_val,y_val


# In[ ]:


############################################################
# Feed Forward Neural Network
############################################################
class FeedForwardNN(nn.Module):
    """ 
        (1) Use self.fc1 as the variable name for your first fully connected layer
        (2) Use self.fc2 as the variable name for your second fully connected layer
    """

    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(64*85*3, 2000)
        self.fc2 = nn.Linear(2000, 5)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out=x
        return out

    """ 
        Please do not change the functions below. 
        They will be used to test the correctness of your implementation 
    """
    def get_fc1_params(self):
        return self.fc1.__repr__()
    
    def get_fc2_params(self):
        return self.fc2.__repr__()


# In[ ]:





# In[ ]:


############################################################
# Convolutional Neural Network
############################################################
class ConvolutionalNN(nn.Module):
    """ 
        (1) Use self.conv1 as the variable name for your first convolutional layer
        (2) Use self.pool1 as the variable name for your first pooling layer
        (3) Use self.conv2 as the variable name for your second convolutional layer
        (4) Use self.pool2 as the variable name for you second pooling layer  
        (5) Use self.fc1 as the variable name for your first fully connected laye
        (6) Use self.fc2 as the variable name for your second fully connected layer
    """
    def __init__(self):
        super(ConvolutionalNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3,stride=1,padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3,stride=1,padding=0)
        self.relu2 = nn.ReLU()
        # Max pool 1
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # Max pool 2
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(8512, 200) 
        # Fully connected 1 (readout)
        self.fc2 = nn.Linear(200, 5)
        
      
    def forward(self, x):
        x=x.numpy()
        x=normalize_image(x)
        out = F.relu(self.conv1(x))
        #print(out.size())

        # Max pool 1
        out = self.pool1(out)
        #print(out.size())

        # Convolution 2 
        out = F.relu(self.conv2(out))
        #print(out.size())

        # Max pool 2 
        out = self.pool2(out)
        #print(out.size())

        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out
      
    """ 
        Please do not change the functions below. 
        They will be used to test the correctness of your implementation
    """
    
    def get_conv1_params(self):
        return self.conv1.__repr__()
    
    def get_pool1_params(self):
        return self.pool1.__repr__()

    def get_conv2_params(self):
        return self.conv2.__repr__()
      
    def get_pool2_params(self):
        return self.pool2.__repr__()
      
    def get_fc1_params(self):
        return self.fc1.__repr__()
    
    def get_fc2_params(self):
        return self.fc2.__repr__()
    


# In[ ]:


#torch.set_default_tensor_type(torch.DoubleTensor)
#torch.set_default_dtype(torch.float32)
def normalize_image(image):
    img=image.copy()
    Rch = img[0,:,:]  # Red color channel
    Gch = img[1,:,:]  # Green color channel
    Bch = img[2,:,:]  # Blue color channel

    Rch_mn = np.mean(Rch)
    Gch_mn = np.mean(Gch)
    Bch_mn = np.mean(Bch)
    Rch_std =np.std(Rch)
    Gch_std =np.std(Gch)
    Bch_std =np.std(Bch)
    
    img[0,:,:]=(Rch-Rch_mn)/Rch_std
    img[1,:,:]=(Gch-Gch_mn)/Gch_std
    img[2,:,:]=(Bch-Bch_mn)/Bch_std
   
    #print(torch.tensor(image).dtype)
    norimg=img
    #norimg=torch.Tensor(img)
    #norimg=T.normalize(image,[Rch_mn, Gch_mn, Bch_mn],[Rch_std,Gch_std ,Bch_std])

    return norimg


# In[ ]:


############################################################
# Optimized Neural Network
############################################################
class OptimizedNN(nn.Module):
    def __init__(self):
        super(OptimizedNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3,stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3,stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size = 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3,stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size = 2)
        self.fc1 = nn.Linear(3072, 200)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(200, 5)
        self.dropout2 = nn.Dropout(p=0.2)
        
    
    def forward(self, x):
        # Normalisation
        x_numpy = x.numpy()
        x = normalize_image(x_numpy)
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = F.relu(self.conv3(out))
        out = self.pool3(out)
        out = out.view(out.size(0),-1)
        out = self.dropout1(out)
        out = F.relu(self.fc1(out))
        out = self.dropout2(out)
        out = self.fc2(out)
        return out
      


# In[ ]:


# dataset=torch.utils.data.TensorDataset(new_x_train,new_y_train)
# validation=torch.utils.data.TensorDataset(x_val,y_val)
# train_loader = DataLoader(dataset, batch_size=64)
# validation_loader = DataLoader(validation, batch_size=64)
# loss_function = nn.CrossEntropyLoss()
# #model=FeedForwardNN()
# model=ConvolutionalNN()
# optimizer = optim.Adagrad(model.parameters(), lr=0.001)
# num_epochs=40
# accuracy,loss ,val_loss=train_val_NN(model,train_loader,validation_loader,loss_function,optimizer,num_epochs)
# #print ("Accuracy: ",np.mean(accuracy), "Training Loss: ", loss, "Valid Loss: ", val_loss)

##testing

#print(torch.tensor(test_loader).dtype)


# In[ ]:



def train_val_NN(neural_network, train_loader, validation_loader, loss_function, optimizer,num_epochs):
    criterion=loss_function
    accuracy = np.zeros(shape=(num_epochs,1))
    val_accuracy = np.zeros(shape=(num_epochs,1))
    loss_np = np.zeros(shape=(num_epochs,1))
    iterator = 0
    for epoch in range(0, num_epochs): ## run the model for 10 epochs
        train_loss = []
        ## training part 
        neural_network.train()
        correct = 0
        acc = 0
        val_acc = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            ## forward propagation
            output = neural_network(data)
            ## loss calculation
            loss = criterion(output, target)
            ## backward propagation
            loss.backward()
            ## weight optimization
            optimizer.step()
            train_loss.append(loss.item())
            ## accuracy computation for each batch
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        acc = correct.item()/len(train_loader.dataset)
        ## evaluation part 
        neural_network.eval()
        correct = 0
        for data, target in validation_loader:
            output = neural_network(data)
            loss = criterion(output, target)
            ## accuracy computation on validation set
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        val_acc=correct.item()/len(validation_loader.dataset)   
        # Add values for each epoch to the arrays
        accuracy[iterator] = acc
        val_accuracy[iterator] = val_acc
        loss_np[iterator] = np.mean(train_loss)
        iterator+=1
    epoch_number = np.arange(0,num_epochs,1)    
    plt.figure()
    plt.plot(epoch_number, accuracy)
    plt.title('training accuracy over epoches')
    plt.xlabel('Number of Epoch=10')
    plt.ylabel('accuracy')
    
    
        # Plot the loss over epoch
    plt.figure()
    plt.plot(epoch_number, loss_np)
    plt.title('loss over epoches')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Loss')
    
    
    plt.figure()
    plt.plot(epoch_number, val_accuracy)
    plt.title('Validation accuracy over epoches')
    plt.xlabel('Number of Epoch')
    plt.ylabel('accuracy')
    #plt.plot(epoch_number, test_accuracy)
#     plt.title('test accuracy over epoches')
    plt.show()
    
    return accuracy,loss_np,val_accuracy

            
        #print ("Epoch:", epoch, "Training Loss: ", np.mean(loss_np), "Valid Loss: ", np.mean(valid_loss))
  #return accuracy,loss_np,val_accuracy


# In[ ]:


def test_NN(neural_network, test_loader):
    model=neural_network
#     model.eval()
#     ## evaluation part 
#     dataiter = iter(test_loader)
#     data,labels = dataiter.next()
#     output = model(data)
#     _, preds_tensor = torch.max(output, 1)
#     preds = np.squeeze(preds_tensor.numpy())
#     print ("Predicted:", preds)
#     print(len(X))
    model.eval()
    test_pred = torch.LongTensor()
    
    for i, data in enumerate(test_loader):
        data = data[0]
        output = model(data)

        pred = output.data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)
        
#         data=data[0]
#         data = Variable(data, volatile=True)            
#         output = model(data)
        
#         pred = output.data.max(1, keepdim=True)[1]
#         test_pred = torch.cat((test_pred, pred), dim=0)


    return test_pred


# In[ ]:


# def train_NN(neural_network, train_loader, loss_function, optimizer,num_epochs):
    
#     for epoch in range(1, num_epochs+1): ## run the model for 10 epochs
#         train_loss = []
#         ## training part 
#         neural_network.train()
#         correct = 0
#         acc = 0
#         val_acc = 0
#         for data, target in train_loader:
#             target=target.long()
#             optimizer.zero_grad()
#             ## forward propagation
#             output = neural_network(data)
#             ## loss calculation
#             loss = loss_function(output, target)
#             ## backward propagation
#             loss.backward()
#             ## weight optimization
#             optimizer.step()
#             train_loss.append(loss.item())

# X,y=extract_data('C:/Users/Rutuja Moharil/hw4_skeleton/HW4/data/images_train.npy','C:/Users/Rutuja Moharil/hw4_skeleton/HW4/data/labels_train.npy')

# #x_train = Dataset(X,y)
# x_train=torch.tensor(X)
# y_train=torch.tensor(y)
# train_dataset = torch.utils.data.TensorDataset(x_train,y_train)
# train_loader = DataLoader(train_dataset, batch_size=32)


# In[ ]:


# Run Baseline FeedForward
# num_epochs=40
# optimized_nn = OptimizedNN()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(optimized_nn.parameters(), lr=0.005)
# train_NN(optimized_nn,train_loader,criterion, optimizer,num_epochs)
# testdata=torch.utils.data.TensorDataset(X_test,y_test)
# test_loader=DataLoader(testdata,batch_size=32)
# test_pred=test_NN(optimized_nn, test_loader)
# #test_pred=test_NN(model,test_loader)


# In[ ]:


# # # Save file 
# test_pred=test_NN(optimized_nn, test_loader)
# with open('HW4_preds.txt', 'w', encoding = 'utf8') as file:
#     for i in range(len(test_loader.dataset)):
#         file.write(str(test_pred[i].item())+'\n')


# In[ ]:


# Run Baseline CNN on Normilized Images


# In[ ]:


# Choose from one of the above models and improve its performance

