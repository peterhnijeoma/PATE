## import required python modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
##import syft
##from syft.frameworks.torch.differential_privacy import pate
##from syft.frameworks.torch.dp import pate
import gc

## required data transformation
data_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,))])
                                     
## load datasets
# ## SVHN training dataset will be used to train teacher models
# train_dataset = datasets.SVHN('datasets/SVHN/train/', split='train',
#                               transform=data_transform,
#                               target_transform=None, download=True)
# ## SVHN test dataset will be used to train student model
# test_dataset = datasets.SVHN('datasets/SVHN/test/', split='test',
#                              transform=data_transform,
#                              target_transform=None, download=True)

## MNIST training dataset will be used to train teacher models
train_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=data_transform)
## MNIST test dataset will be used to train student model
test_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=data_transform)

##global variables
number_teachers = 250 #200 #150 #120 #100 #50 #10           
data_batch_size = 32 
epsilon = 0.2
beta = 1 / epsilon
proc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## functions

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        ##self.fc0 = nn.Linear(3072, 1024)
        #self.fc0 = nn.Linear(2352, 1568)
        #self.fc1 = nn.Linear(1568, 784)
        self.fc2 = nn.Linear(784, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 10)

        self.fc_drop = nn.Dropout(p=0.1)  ## 0.2
    
    def forward(self, img):
        ## flatten input image
        img_flat = img.view(img.shape[0], -1)
        #img_flat = F.relu(self.fc0(img_flat))
        #img_flat = F.relu(self.fc1(img_flat))
        #img_flat = self.fc_drop(img_flat)
        img_flat = F.relu(self.fc2(img_flat))
        img_flat = F.relu(self.fc3(img_flat))
        img_flat = self.fc_drop(img_flat)
        img_flat = F.relu(self.fc4(img_flat))
        img_flat = F.relu(self.fc5(img_flat))
        return F.log_softmax(self.fc6(img_flat), dim=1)

## training variables
criterion = nn.NLLLoss()
epochs = 100 ##60 

def partition_dataset(train_dataset, number_teachers):
    """ Function to partition dataset for teacher classifiers.
        The data will be divided equally by the number of teachers
        using integer division. """

    teacher_datasets = []
    # print()
    # print("length of dataset is: ", len(train_dataset))
    subdataset_size = len(train_dataset) // number_teachers
    # print()
    # print("each of 10 teacher data set size is: ", subdataset_size)
    remainder_size = len(train_dataset) % number_teachers
    # print()
    # print("remainder size is: ", remainder_size)
    ##print(1 if (remainder_size > 0))
    
    for i in range(number_teachers):
        if (remainder_size > 0):
            index_range = list(range(i*subdataset_size, ((i+1)*subdataset_size) + 1))
        else:
            index_range = list(range(i*subdataset_size, ((i+1)*subdataset_size)))
        remainder_size -= 1
        # print()
        # print("length of index range is: ", len(index_range))
        # print("teacher {} index range is {}".format(i, index_range))
        teacher_data_subset = Subset(train_dataset, index_range)
        loader = torch.utils.data.DataLoader(teacher_data_subset, batch_size=data_batch_size, shuffle=True)
        teacher_datasets.append(loader)
        
    return teacher_datasets

def train_model(model, train_loader, optimizer):
    """ Function to train a model. It will be used to train both
        teachers and the student models. """

    running_loss = 0
    for e in range(epochs):
        ## ensure model is in training mode
        model.train()
              
        for img, label in train_loader:
            ## print(img.shape)
            ## move the data to the appropriate device
            if torch.cuda.is_available():
                img, label = img.cuda(), label.cuda()
                ##torch.cuda.empty_cache() ## trying to solve GPU out of memory issue
                ##gc.collect()
            optimizer.zero_grad()
            result = model.forward(img)
            train_loss = criterion(result, label)
            train_loss.backward()
            optimizer.step()

            running_loss += train_loss.item()

    return running_loss


def make_prediction(model, data_loader):
    """ Function for making inference. This function will be used
        to label the data set for student training and also used to vaerify
        the accuracy of the student model. """

    ## ensure the model is in evaluation mode
    model.eval()
    outputs = []  ##torch.zeros(0, dtype=torch.long)
    ##print("outputs shape is: ", outputs)
    
    for img, labels in data_loader:
        if torch.cuda.is_available():
            img, labels = img.cuda(), labels.cuda()
        with torch.no_grad():
            output = model.forward(img)
            label = torch.argmax(torch.exp(output), dim=1)
        ##print("the shape of label is: ", label.shape)
        outputs = np.append(outputs, label.tolist())
        ##print("label is: ", label.tolist())
    ##print("outputs before single array conversion: ", outputs)
    ##outputs = np.array([np.array(o) for o in outputs])
    ##print("outputs after single array conversion: ", outputs)
    ##print("outputs converted to integer is: ",outputs.astype(int))
        
    return outputs.astype(int)

def aggregate_teachers(teachers, dataloader):
    """ Function to aggregate teachers' label to one label and add noise
        to make it Differentially Private. """
    
    ## get predictions from each teacher model
    preds = [] ##torch.zeros((number_teachers, len(dataloader)))
    ##print("declare shape for preds is: ",preds.shape)
    for i, model in enumerate(teachers):
        results = make_prediction(model, dataloader)
        ##print("the shape of prediction results is: ", results.shape)
        # print()
        # print("results - student labels are: ", results)
        preds.append(results)
        ##print("after prediction preds shape is: ", preds.shape)
    # print()
    # print("preds is: ", preds)
    # print()
    # print("transposed preds is: ", np.transpose(preds))
    
    ## aggregate the predictions for each sample data
    labels = np.array([]).astype(int)
    for image_preds in np.transpose(preds):
        label_counts = np.bincount(image_preds, minlength=10)

        for i in range(len(label_counts)):
            ## add noise - DP
            label_counts[i] += np.random.laplace(0, beta, 1)
        
        ## aggregate label
        new_label = np.argmax(label_counts)
        labels = np.append(labels, new_label)
        # print()
        ##print("student data lables are: ", labels)
        # print("student lables length is: ", len(labels))
    
    return preds, labels

def student_loader(student_train_loader, labels):
    """ Function - a generator, for merging student training data with
        aggregated labels from teachers """

    for i, (data, _) in enumerate(iter(student_train_loader)):
        yield data, torch.from_numpy(labels[i*len(data): (i+1)*len(data)])



## start of execution

print("the current device is: ", proc_device)

## partition data per number of teachers
teacher_train_datasets = partition_dataset(train_dataset, number_teachers)
## divide the student dataset into training and testing partitions - 90% training
## and 10% testing.
dataset_size = len(test_dataset)
student_train_dataset = Subset(test_dataset, list(range((dataset_size//100)*90)))
student_test_dataset = Subset(test_dataset, list(range((dataset_size//100)*90+1, dataset_size)))
student_train_loader = torch.utils.data.DataLoader(student_train_dataset, batch_size=data_batch_size)
student_test_loader = torch.utils.data.DataLoader(student_test_dataset, batch_size=data_batch_size)

## train each teacher
teacher_models = []
for i in range(number_teachers):
    model = Classifier()
    model = model.to(proc_device)  ## move model to GPU or CPU - use GPU if available
    teacher_optimizer = optim.Adam(model.parameters(), lr=0.001)
    teacher_trn_loss = train_model(model, teacher_train_datasets[i], teacher_optimizer)
    ##model = model.to("cpu")
    teacher_models.append(model)
    ##torch.cuda.empty_cache()
    ##gc.collect()
    ##print("finished training for teacher ", i)

print("done with training teachers")
print()

## aggregate teacher labels
predictions, studt_trn_data_labels = aggregate_teachers(teacher_models, student_train_loader)

## check epsilon is acceptable - PATE Analysis
##data_dep_epsilon, data_indep_epsilon = pate.perform_analysis(teacher_preds=predictions, indices=studt_trn_data_labels, noise_eps=epsilon, delta=1e-5)
##print("Data Independent Epsilon:", data_indep_epsilon)
##print("Data Dependent Epsilon:", data_dep_epsilon)
##print()

## train student model
student_model = Classifier()
student_model = student_model.to(proc_device)  ## move model to GPU or CPU - use GPU if available
studt_optimizer = optim.Adam(student_model.parameters(), lr=0.001)
student_trn_loader = student_loader(student_train_loader, studt_trn_data_labels)

##steps = 0
running_loss = 0
for e in range(epochs):
    running_loss = 0
    ## ensure model is in train mode
    student_model.train()
    ## merge data and labels
    ##train_loader = student_loader(student_train_loader, studt_trn_data_labels)
    for imgs, labels in student_train_loader:   ##train_loader:
        ##steps += 1 ## new set of sample
        if torch.cuda.is_available():
            imgs, labels = imgs.cuda(), labels.cuda()
        studt_optimizer.zero_grad()
        output = student_model.forward(imgs)
        loss = criterion(output, labels)
        loss.backward()
        studt_optimizer.step()
        running_loss += loss.item()
        
# student_model.train()
# running_loss = train_model(student_model, student_trn_loader, studt_optimizer)

## model is in evaluation mode and gradients not stored
student_model.eval()
test_loss = 0
accuracy = 0
with torch.no_grad():
    for images, labels in student_test_loader:
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        log_ps = student_model(images)
        test_loss += criterion(log_ps, labels).item()
                    
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))

## back to train mode
student_model.train()

##print("Epoch: {}/{}.. ".format(e+1, epochs),
print("Average Train Loss: {:.3f}.. ".format(running_loss/len(student_train_loader)),
      "Average Test Loss: {:.3f}.. ".format(test_loss/len(student_test_loader)),
      "Average Test Accuracy: {:.3f}".format(accuracy))


