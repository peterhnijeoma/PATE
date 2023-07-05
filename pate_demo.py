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
import sys
import time

## required data transformation
if sys.argv[4] == 'SVHN':  # CL argument 4 is for the dataset to use
    data_transform = transforms.Compose([transforms.CenterCrop(28),
                                         ##transforms.Grayscale(),
                                         transforms.ToTensor(),
                                         ##transforms.Normalize((0.5,), (0.5,))
                                         transforms.Normalize([0.5, 0.5, 0.5], 
                                                              [0.5, 0.5, 0.5])
                                        ])

    ## load datasets
    ## SVHN training dataset will be used to train teacher models
    train_dataset = datasets.SVHN('datasets/SVHN/train/', split='train',
                                  transform=data_transform,
                                  target_transform=None, download=True)
    ## SVHN test dataset will be used to train student model
    test_dataset = datasets.SVHN('datasets/SVHN/test/', split='test',
                                 transform=data_transform,
                                 target_transform=None, download=True)
    
    # print("SVHN test datset samples are:", len(test_dataset))
    # print()

    epochs = 100 ##200
    
    class Classifier(nn.Module):
        # def __init__(self):
        #     super().__init__()
        #     ##self.fc0 = nn.Linear(3072, 1024)
        #     # self.fc0 = nn.Linear(3072, 2048)
        #     # self.fc1 = nn.Linear(2048, 1024)
        #     # self.fc2 = nn.Linear(1024, 784)
        #     self.fc3 = nn.Linear(784, 512)
        #     self.fc4 = nn.Linear(512, 256)
        #     self.fc5 = nn.Linear(256, 128)
        #     self.fc6 = nn.Linear(128, 64)
        #     self.fc7 = nn.Linear(64, 10)

        #     self.fc_drop = nn.Dropout(p=0.2)  ## 0.2

        # def __init__(self):
        #     super().__init__()
        #     ##self.fc0 = nn.Linear(3072, 1024)
        #     self.fc0 = nn.Linear(3072, 2048)
        #     self.fc1 = nn.Linear(2048, 1024)
        #     self.fc2 = nn.Linear(1024, 784)
        #     self.fc3 = nn.Linear(784, 512)
        #     self.fc4 = nn.Linear(512, 256)
        #     self.fc5 = nn.Linear(256, 128)
        #     self.fc6 = nn.Linear(128, 64)
        #     self.fc7 = nn.Linear(64, 10)

        #     self.fc_drop = nn.Dropout(p=0.2)  ## 0.2

        ## convolution
        def __init__(self):
            super().__init__()

            self.conv_1 = nn.Conv2d(3, 16, 5, 1, 2)
            ##self.conv_1 = nn.Conv2d(1, 16, 5, 1, 2)
            self.batch_1 = nn.BatchNorm2d(16)
            self.pool = nn.MaxPool2d(2,2)
            self.conv_2 = nn.Conv2d(16, 32, 5, 1, 2)
            self.batch_2 = nn.BatchNorm2d(32)
            self.fc1 = nn.Linear(32*7*7, 64)
            ##self.fc1 = nn.Linear(32*8*8, 64)
            self.fc2 = nn.Linear(64, 10)

            self.fc_drop = nn.Dropout(p=0.2)
            self.conv_drop = nn.Dropout2d()
    
        # def forward(self, img):
        #     ## flatten input image
        #     img_flat = img.view(img.shape[0], -1)
        #     # img_flat = F.relu(self.fc0(img_flat))
        #     # img_flat = self.fc_drop(img_flat)
        #     # img_flat = F.relu(self.fc1(img_flat))
        #     # img_flat = self.fc_drop(img_flat)
        #     # img_flat = F.relu(self.fc2(img_flat))
        #     # img_flat = self.fc_drop(img_flat)
        #     img_flat = F.relu(self.fc3(img_flat))
        #     img_flat = self.fc_drop(img_flat)
        #     img_flat = F.relu(self.fc4(img_flat))
        #     img_flat = self.fc_drop(img_flat)
        #     img_flat = F.relu(self.fc5(img_flat))
        #     img_flat = self.fc_drop(img_flat)
        #     img_flat = F.relu(self.fc6(img_flat))
        #     return F.log_softmax(self.fc7(img_flat), dim=1)

        # def forward(self, img):
        #     ## flatten input image
        #     img_flat = img.view(img.shape[0], -1)
        #     img_flat = F.relu(self.fc0(img_flat))
        #     img_flat = self.fc_drop(img_flat)
        #     img_flat = F.relu(self.fc1(img_flat))
        #     img_flat = self.fc_drop(img_flat)
        #     img_flat = F.relu(self.fc2(img_flat))
        #     img_flat = self.fc_drop(img_flat)
        #     img_flat = F.relu(self.fc3(img_flat))
        #     img_flat = self.fc_drop(img_flat)
        #     img_flat = F.relu(self.fc4(img_flat))
        #     img_flat = self.fc_drop(img_flat)
        #     img_flat = F.relu(self.fc5(img_flat))
        #     img_flat = self.fc_drop(img_flat)
        #     img_flat = F.relu(self.fc6(img_flat))
        #     return F.log_softmax(self.fc7(img_flat), dim=1)

        ## use with convolution
        def forward(self, img):
            img_out = self.conv_1(img)
            img_out = self.conv_drop(img_out)
            ##print("image shape after conv 1 ", img_out.shape)
            img_out = F.relu(self.batch_1(img_out))
            ##print("image shape after conv 1 and batch 1", img_out.shape)
            img_out = self.pool(img_out)

            ##print("image shape after conv 1, batch 1, and pool", img_out.shape)

            img_out = self.conv_2(img_out)
            ##img_out = self.conv_drop(img_out)
            ##print("image shape after conv 2", img_out.shape)
            img_out = F.relu(self.batch_2(img_out))
            ##print("image shape after conv 2 and batch 2", img_out.shape)
            img_out = self.pool(img_out)

            ##print("image shape after conv 2, batch 2, and pool", img_out.shape)

            # flatten img_out for FC layers
            img_flat = img_out.view(img_out.size(0), -1)

            img_flat = self.fc1(img_flat)
            img_flat = self.fc_drop(img_flat)
            img_flat = self.fc2(img_flat)

            return F.log_softmax(img_flat, dim=1)
elif sys.argv[4] == 'FMNIST':  # CL argument 4 is for the dataset to use
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5], [0.5])
                                        ])

    ## load datasets
    ## fashion MNIST training dataset will be used to train teacher models
    train_dataset = datasets.FashionMNIST('./fashionMNIST/', train=True,
                                          download=True,
                                          transform=data_transform)
    ## fashion MNIST test dataset will be used to train student model
    test_dataset = datasets.FashionMNIST('./fashionMNIST/', train=False,
                                         download=True,
                                         transform=data_transform)
    
    # print("fashion MNIST test datset samples are:", len(test_dataset))
    # print()

    epochs = 100 ##200
    
    class Classifier(nn.Module):
        ## convolution
        def __init__(self):
            super().__init__()

            self.conv_1 = nn.Conv2d(1, 16, 5, 1, 2)
            self.batch_1 = nn.BatchNorm2d(16)
            self.pool = nn.MaxPool2d(2,2)
            self.conv_2 = nn.Conv2d(16, 32, 5, 1, 2)
            self.batch_2 = nn.BatchNorm2d(32)
            self.fc1 = nn.Linear(32*7*7, 64)
            self.fc2 = nn.Linear(64, 10)

            self.fc_drop = nn.Dropout(p=0.2)
            self.conv_drop = nn.Dropout2d()

        ## use with convolution
        def forward(self, img):
            img_out = self.conv_1(img)
            img_out = self.conv_drop(img_out)
            ##print("image shape after conv 1 ", img_out.shape)
            img_out = F.relu(self.batch_1(img_out))
            ##print("image shape after conv 1 and batch 1", img_out.shape)
            img_out = self.pool(img_out)

            ##print("image shape after conv 1, batch 1, and pool", img_out.shape)

            img_out = self.conv_2(img_out)
            ##img_out = self.conv_drop(img_out)
            ##print("image shape after conv 2", img_out.shape)
            img_out = F.relu(self.batch_2(img_out))
            ##print("image shape after conv 2 and batch 2", img_out.shape)
            img_out = self.pool(img_out)

            ##print("image shape after conv 2, batch 2, and pool", img_out.shape)

            # flatten img_out for FC layers
            img_flat = img_out.view(img_out.size(0), -1)

            img_flat = self.fc1(img_flat)
            img_flat = self.fc_drop(img_flat)
            img_flat = self.fc2(img_flat)

            return F.log_softmax(img_flat, dim=1)
        
else:  # MNIST
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])
    ## MNIST training dataset will be used to train teacher models
    train_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=data_transform)
    ## MNIST test dataset will be used to train student model
    test_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=data_transform)

    # print("MNIST test datset samples are:", len(test_dataset))
    # print()

    epochs = 100 ##200  ##60 ##10

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc2 = nn.Linear(784, 512)
            self.fc3 = nn.Linear(512, 256)
            self.fc4 = nn.Linear(256, 128)
            self.fc5 = nn.Linear(128, 64)
            self.fc6 = nn.Linear(64, 10)

            self.fc_drop = nn.Dropout(p=0.2)  ## 0.2
    
        def forward(self, img):
            ## flatten input image
            img_flat = img.view(img.shape[0], -1)
            img_flat = F.relu(self.fc2(img_flat))
            img_flat = self.fc_drop(img_flat)
            img_flat = F.relu(self.fc3(img_flat))
            img_flat = self.fc_drop(img_flat)
            img_flat = F.relu(self.fc4(img_flat))
            img_flat = self.fc_drop(img_flat)
            img_flat = F.relu(self.fc5(img_flat))
            return F.log_softmax(self.fc6(img_flat), dim=1)
        
##global variables
number_teachers = int(sys.argv[1]) # get number of teachers from CL argument 1 ##10 ###250 #200 #150 #120 #100 #50 #10     
data_batch_size = 64 ##8 ##16 ##32    
epsilon = float(sys.argv[2]) # get epsilon from CL argument 2 ##0.2
if sys.argv[3] == "true":
    beta = 1 / epsilon
else:
    beta = 0  ## when no noise is added, the value of beta does not matter

##gpu_1_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multi_gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##print("single gpu device is:", gpu_1_device)
##print("multiple gpu is:", multi_gpu_device)

## training variables
criterion = nn.NLLLoss()

## functions
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
            # if torch.cuda.is_available():
            #     img, label = img.cuda(), label.cuda()
            #     ##torch.cuda.empty_cache() ## trying to solve GPU out of memory issue
            #     ##gc.collect()

            ## move data to appropriate GPU
            gpu_img, gpu_label = img.to(multi_gpu_device), label.to(multi_gpu_device)
            optimizer.zero_grad()
            result = model.forward(gpu_img)   ##model.forward(img)
            train_loss = criterion(result, gpu_label).to(multi_gpu_device)  ##criterion(result, label)
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
        ##if torch.cuda.is_available():
        ##img, labels = img.cuda(), labels.cuda()
        gpu_img, gpu_labels = img.to(multi_gpu_device), labels.to(multi_gpu_device)
        with torch.no_grad():
            output = model.forward(gpu_img)
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
            if (sys.argv[3] == 'true'):
                label_counts[i] += np.random.laplace(0, beta, 1)
            else:
                label_counts[i] += 0  ## add 0 noise to count
        
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

# print("the current device is: ", multi_gpu_device) ## proc_device)
# print("epochs is:", epochs)

start_time = time.time()
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
    ## use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        ##print("training teacher model number", i,"- will use", torch.cuda.device_count(), "GPUs.")
        model = nn.DataParallel(model)
    ## move model to appropriate GPU or CPU - use GPU if available
    model = model.to(multi_gpu_device) ##proc_device)
    teacher_optimizer = optim.Adam(model.parameters(), lr=0.003)
    teacher_trn_loss = train_model(model, teacher_train_datasets[i], teacher_optimizer)
    ##model = model.to("cpu")
    teacher_models.append(model)
    ##torch.cuda.empty_cache()
    ##gc.collect()
    ##print("finished training for teacher", i)

# print("done with training teachers")
# print()

## aggregate teacher labels
predictions, studt_trn_data_labels = aggregate_teachers(teacher_models, student_train_loader)

# print("done with aggregation")
# print()

## check epsilon is acceptable - PATE Analysis
##data_dep_epsilon, data_indep_epsilon = pate.perform_analysis(teacher_preds=predictions, indices=studt_trn_data_labels, noise_eps=epsilon, delta=1e-5)
##print("Data Independent Epsilon:", data_indep_epsilon)
##print("Data Dependent Epsilon:", data_dep_epsilon)
##print()

## train student model
student_model = Classifier()
if torch.cuda.device_count() > 1:
    ##print("training student model - will use", torch.cuda.device_count(), "GPUs.")
    student_model = nn.DataParallel(student_model)
student_model = student_model.to(multi_gpu_device)  ## move model to GPU or CPU - use GPU if available
studt_optimizer = optim.Adam(student_model.parameters(), lr=0.003)

## merge student training data and labels from teacher aggregation
student_trn_loader = student_loader(student_train_loader, studt_trn_data_labels)

## Student model training
for e in range(epochs):
    running_loss = 0
    for imgs, labels in student_trn_loader:
        ##if torch.cuda.is_available():
        gpu_imgs, gpu_labels = imgs.to(multi_gpu_device), labels.to(multi_gpu_device)
        studt_optimizer.zero_grad()

        output = student_model.forward(gpu_imgs)
        loss = criterion(output, gpu_labels).to(multi_gpu_device)
        loss.backward()
        studt_optimizer.step()

        running_loss += loss.item()

# print("Done training Student model")
# print()

## check accuracy of student model
## model is in evaluation mode and gradients not stored
student_model.eval()
# test_loss = 0
# accuracy = 0
##iteration = 0
##print("number of student test samples is:", len(student_test_loader))
##print("number of student test dataset samples is:", len(student_test_dataset))
##print()
with torch.no_grad():
    tests_accuracy = []
    tests_loss = []
    for k in range(10):
        test_loss = 0
        accuracy = 0
        for images, labels in student_test_loader:
            ##iteration += 1
            ##if torch.cuda.is_available():
            gpu_images, gpu_labels = images.to(multi_gpu_device), labels.to(multi_gpu_device)
            log_ps = student_model(gpu_images)
            test_loss += criterion(log_ps, gpu_labels).item()
                        
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == gpu_labels.view(*top_class.shape)

            ##print("number of images processed for iteration " + str(iteration) + " is:", images.size())
            ##print("sum of equals is:", equals.sum().item())
            accuracy += equals.sum().item()
            ##print("part accuracy is", accuracy)
        
        tests_accuracy.append(accuracy/len(student_test_dataset))
        tests_loss.append(test_loss/len(student_test_loader))

## back to train mode
student_model.train()

end_time = time.time()

##print("Epoch: {}/{}.. ".format(e+1, epochs),
print("Average Train Loss: {:.3f}.. ".format(running_loss/len(student_train_loader)),
      "Average Test Loss for 10 runs: {:.3f}.. ".format(sum(tests_loss) / len(tests_loss)),
      "Average Test Accuracy for 10 runs: {:.3f}.. ".format(sum(tests_accuracy)/len(tests_accuracy)),
      "Total Elapsed time is: {:.3f} minutes".format((end_time - start_time)/60))
# print("Average Train Loss: {:.3f}.. ".format(running_loss/len(student_train_loader)),
#       "Average Test Loss: {:.3f}.. ".format(test_loss/len(student_test_loader)),
#       "Average Test Accuracy: {:.3f}.. ".format(accuracy/len(student_test_dataset)),
#       "Elapsed time is: {:.3f} minutes".format((end_time - start_time)/60))


