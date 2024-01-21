import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import classification_report
from scipy.stats import entropy
from time import time
from torch.utils.data import random_split

num_classes = 10
num_layers = 5

device = torch.device("cpu")

torch.manual_seed(42)
dataset = CIFAR10(root="./data", download=True, transform=ToTensor())
test_dataset = CIFAR10(root="./data", train=False, transform=ToTensor())

batch_size = 128
val_size = 5000
train_size = len(dataset) - val_size
_, val_ds = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_ds, batch_size * 2, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size * 2, num_workers=4)


class Branch(nn.Module):
    def __init__(self, in_channels, in_features):
        super(Branch, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, stride=2
        )
        self.bn = nn.BatchNorm2d(num_features=16)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.bn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.in_channels = [32, 32, 64, 64, 128]
        self.in_features = [3600, 784, 784, 144, 144]
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding="same"
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding="same"
        )
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding="same"
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding="same"
        )
        self.conv6 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding="same"
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.bn5 = nn.BatchNorm2d(num_features=128)
        self.bn6 = nn.BatchNorm2d(num_features=128)
        self.dropout3 = nn.Dropout(p=0.4)

        self.branch1 = Branch(
            in_channels=self.in_channels[0], in_features=self.in_features[0]
        )
        self.branch2 = Branch(
            in_channels=self.in_channels[1], in_features=self.in_features[1]
        )
        self.branch3 = Branch(
            in_channels=self.in_channels[2], in_features=self.in_features[2]
        )
        self.branch4 = Branch(
            in_channels=self.in_channels[3], in_features=self.in_features[3]
        )
        self.branch5 = Branch(
            in_channels=self.in_channels[4], in_features=self.in_features[4]
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=2048, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=128)
        self.bn7 = nn.BatchNorm1d(num_features=128)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=128, out_features=num_classes)

        self.num_layers = num_layers

    def forward(self, tensor_after_previous_layer, exit_layer_idx=num_layers):
        if exit_layer_idx == 0:
            x = self.conv1(tensor_after_previous_layer)
            x = F.relu(x)
            tensor_after_layer = self.bn1(x)
            predicted_scores_from_layer = self.branch1(tensor_after_layer)

        elif exit_layer_idx == 1:
            x = self.conv2(tensor_after_previous_layer)
            x = F.relu(x)
            x = self.bn2(x)
            x = self.pool1(x)
            tensor_after_layer = self.dropout1(x)
            predicted_scores_from_layer = self.branch2(tensor_after_layer)

        elif exit_layer_idx == 2:
            x = self.conv3(tensor_after_previous_layer)
            x = F.relu(x)
            tensor_after_layer = self.bn3(x)
            predicted_scores_from_layer = self.branch3(tensor_after_layer)

        elif exit_layer_idx == 3:
            x = self.conv4(tensor_after_previous_layer)
            x = F.relu(x)
            x = self.bn4(x)
            x = self.pool2(x)
            tensor_after_layer = self.dropout2(x)
            predicted_scores_from_layer = self.branch4(tensor_after_layer)

        elif exit_layer_idx == 4:
            x = self.conv5(tensor_after_previous_layer)
            x = F.relu(x)
            tensor_after_layer = self.bn5(x)
            predicted_scores_from_layer = self.branch5(tensor_after_layer)

        elif exit_layer_idx == 5:
            x = self.conv6(tensor_after_previous_layer)
            x = F.relu(x)
            x = self.bn6(x)
            x = self.pool3(x)
            x = self.dropout3(x)

            x = self.flatten(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            x = F.relu(x)
            x = self.fc4(x)
            x = F.relu(x)
            x = self.bn7(x)
            tensor_after_layer = self.dropout4(x)
            predicted_scores_from_layer = self.fc5(tensor_after_layer)

        else:
            ValueError(f"exit_layer_idx {exit_layer_idx} should be int within 0 to 5")

        return tensor_after_layer, predicted_scores_from_layer


model = Baseline().to(device)
model.load_state_dict(torch.load("cifar10_branchyNet_m.h5", map_location="cpu"))
model.eval()




def cutoff_exit_performance_check(cutoff, print_per_layer_performance=False):
    """TODO: On test data, run the model by iterating through exit layer indices.
    Decide, based on entropy, whether to exit from a particular layer or not.
    Please utilize tensors after a layer for the next layer, if not exited.
    If print_per_layer_performance is True, please print accuracy and time
    for each layer. We want to see the printables for only one value. When
    plotting, you don't need to print these.
    """
    total_correct = 0
    total_samples = 0
    total_time = 0.0
    layerwise_accuracies = []
    layerwise_time=[]
    layerwise_exit_count =[]



    with torch.no_grad():

        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            batch_time_li=[]
            batch_accuracy_li=[]
            batch_exit_count_li=[]

            # print(inputs.shape)
            tensor_after_layer = inputs
            samples_remaining = torch.arange(inputs.size(0))  # Indices of samples that haven't exited
           

            for exit_layer_idx in range(num_layers + 1):
                start_time = time()
                tensor_after_layer, predicted_scores_from_layer = model(
                    tensor_after_layer, exit_layer_idx
                )
                # print(predicted_scores_from_layer.shape)
                end_time = time()
                total_time += end_time - start_time
                

                # if exit_layer_idx < num_layers:
                # Calculate entropy for each remaining sample
                # print('Here')
                # print(samples_remaining.shape)
                probs = F.softmax(predicted_scores_from_layer[samples_remaining], dim=1)
                # print(probs.shape)
                entropies = entropy(probs,axis=1)
                

                # Decide whether to exit each sample based on cutoff
                should_exit = entropies > cutoff

                if(exit_layer_idx<num_layers):
                    samples_in_exit = samples_remaining[should_exit]
                    samples_remaining = samples_remaining[~should_exit]
                else:
                    samples_in_exit=samples_remaining


                # print(samples_in_exit)
                # Calculate accuracy for the current layer
                layer_correct = (predicted_scores_from_layer[samples_in_exit].argmax(dim=1) == labels[samples_in_exit]).sum().item()
                total_correct += layer_correct
                total_samples += len(samples_in_exit)
                layer_accuracy = layer_correct / len(samples_in_exit) if len(samples_in_exit) else 0
                batch_accuracy_li.append(layer_accuracy)
                batch_time_li.append(end_time-start_time)
                batch_exit_count_li.append(len(should_exit))

            layerwise_accuracies.append(batch_accuracy_li)
            layerwise_time.append(batch_time_li)
            layerwise_exit_count.append(batch_exit_count_li)

            # _, predicted = torch.max(predicted_scores_from_layer[samples_remaining], 1)
            # total_samples += len(samples_remaining)
            # total_correct += (predicted == labels[samples_remaining]).sum().item()

    # overall_accuracy = total_correct / total_samples

    layerwise_accuracies_np=np.array(layerwise_accuracies)
    layerwise_time_np = np.array(layerwise_time)
    layerwise_exit_count_np = np.array(layerwise_exit_count)

    layerwise_accuracies_np=np.mean(layerwise_accuracies_np,axis=0,keepdims=True)
    layerwise_time_np = np.sum(layerwise_time_np,axis=0,keepdims=True)

    layerwise_exit_count_np=np.sum(layerwise_exit_count_np,axis=0,keepdims=True)

    weighted_average = np.sum(layerwise_accuracies_np * layerwise_exit_count_np)/np.sum(layerwise_exit_count_np)

    if print_per_layer_performance:

        print("The list denotes the accurary and time at each exit_index")        
        print(f"Layerwise Accuracy:",layerwise_accuracies_np)
        print(f"LayerWise Time:",layerwise_time_np)

    return weighted_average, total_time
    

def estimate_thresholds(desired_accuracy):
    """
    TODO: On validation data, for each layer, estimate entropy cutoff that
    gives the desired accuracy. Consider the samples exited and skip those
    samples when estimating the thresholds for the following layers.
    """
    estimated_thresholds = []

    with torch.no_grad():
        for exit_layer_idx in range(num_layers + 1):
            total_correct = 0
            total_samples = 0
            # layer_accuracies = []

            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                tensor_after_layer = inputs
                # samples_remaining = torch.arange(inputs.size(0))

                for current_layer_idx in range(exit_layer_idx + 1):
                    tensor_after_layer, predicted_scores_from_layer = model(
                        tensor_after_layer, current_layer_idx
                    )

                    if current_layer_idx == exit_layer_idx:
                        probs = F.softmax(
                            predicted_scores_from_layer, dim=1
                        )

                        total_correct+= (torch.argmax(probs,1)==labels).sum().item()
                        total_samples+=len(labels)

                        
            

            overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
            if overall_accuracy > desired_accuracy:
                estimated_thresholds.append(0.0)

            else:
                val_entropies=[]
                for batch in val_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)

                    tensor_after_layer = inputs

                    for current_layer_idx in range(exit_layer_idx + 1):
                        tensor_after_layer, predicted_scores_from_layer = model(
                            tensor_after_layer, current_layer_idx
                        )

                        if current_layer_idx == exit_layer_idx:
                            probs = F.softmax(
                                predicted_scores_from_layer, dim=1
                            )


                            entropies = entropy(probs,axis=1)
                            val_entropies.extend(entropies)
                
                sorted_val_entropies=np.sort(val_entropies)
                threshold_index= int(1-desired_accuracy*len(sorted_val_entropies))
                threshold = sorted_val_entropies[threshold_index]

                estimated_thresholds.append(threshold)



            # # Adjust the threshold to achieve the desired accuracy
            # estimated_threshold = np.percentile(
            #     np.concatenate(layer_accuracies), (1 - desired_accuracy) * 100
            # )
            # estimated_thresholds.append(estimated_threshold)

    return estimated_thresholds


# 1(a) For a fixed value of cutoff, show performance for all layers.
fixed_cutoff = 0.6
overall_accuracy_fixed, total_time_fixed = cutoff_exit_performance_check(
    cutoff=fixed_cutoff, print_per_layer_performance=True
)
print(f"\nOverall Accuracy (Fixed Cutoff {fixed_cutoff}): {overall_accuracy_fixed:.4f}")
print(f"Total Inference Time: {total_time_fixed:.4f} seconds")


# 1(b) Plot overall accuracy vs cutoff, total time vs cutoff, and total time vs overall accuracy.
# 2.3 is chosen since log 10. But I have chosen until 2.7
cutoff_values =  np.random.uniform(0, 2.7, 100)
cutoff_values.sort()
# print(cutoff_values)
accuracy_values = []
time_values = []

for cutoff in cutoff_values:
    overall_accuracy, total_time = cutoff_exit_performance_check(cutoff)
    accuracy_values.append(overall_accuracy)
    time_values.append(total_time)

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(cutoff_values, accuracy_values, marker='o')
plt.xlabel('Cutoff')
plt.ylabel('Overall Accuracy')
plt.title('Overall Accuracy vs Cutoff')

plt.subplot(1, 3, 2)
plt.plot(cutoff_values, time_values, marker='o')
plt.xlabel('Cutoff')
plt.ylabel('Total Inference Time (s)')
plt.title('Total Inference Time vs Cutoff')

plt.subplot(1, 3, 3)
plt.plot(accuracy_values, time_values, marker='o')
plt.xlabel('Overall Accuracy')
plt.ylabel('Total Inference Time (s)')
plt.title('Total Inference Time vs Overall Accuracy')

plt.tight_layout()
plt.savefig('figure1.png')


# # 2(a) On validation data, estimate threshold for each layer based on desired minimum accuracy.
desired_accuracy = 0.80
estimated_thresholds = estimate_thresholds(desired_accuracy)
print(f"Estimated Thresholds for Desired Accuracy {desired_accuracy}: {estimated_thresholds}")

# 2(c) Vary the desired minimum accuracy and generate lists of thresholds.
desired_accuracies = [0.70,0.75,0.80]
thresholds_lists = []

for acc in desired_accuracies:
    thresholds_lists.append(estimate_thresholds(acc))

# Plot total time vs overall accuracy for different threshold lists
plt.figure(figsize=(10, 8))

max_accuracy=0

cutoff_exit_performance_check_dict={}
for idx, thresholds in enumerate(thresholds_lists):
    overall_accuracies = []
    total_times = []

    for cutoff in thresholds:
        if cutoff not in cutoff_exit_performance_check_dict:
            overall_accuracy, total_time = cutoff_exit_performance_check(cutoff)
            cutoff_exit_performance_check_dict[cutoff]=[overall_accuracy,total_time]

        overall_accuracy, total_time = cutoff_exit_performance_check_dict[cutoff]
        
        overall_accuracies.append(overall_accuracy)
        if(overall_accuracy>max_accuracy):
            max_accuracy=overall_accuracy
        total_times.append(total_time)

    plt.plot(total_times, overall_accuracies, label=f'Desired Accuracy = {desired_accuracies[idx]:.2f}')

plt.xlabel('Total Inference Time (s)')
plt.ylabel('Overall Accuracy')
plt.title('Total Inference Time vs Overall Accuracy for Different Threshold Lists')
plt.legend()
plt.savefig('figure2.png')

best_threshold=0
best_time=float('inf')
for idx, thresholds in enumerate(thresholds_lists):
    
    for cutoff in thresholds:
        overall_accuracy, total_time = cutoff_exit_performance_check_dict[cutoff]
        if overall_accuracy == max_accuracy and total_time<best_time:
            best_threshold=cutoff
            best_time=total_time

# Evaluate accuracy and inference time on test data using the best threshold
test_accuracy, test_inference_time = cutoff_exit_performance_check(best_threshold, print_per_layer_performance=True)

print(f'Best Threshold is :{best_threshold}')
print(f'Test Accuracy using the Best Threshold: {test_accuracy}')
print(f'Inference Time on Test Data using the Best Threshold: {test_inference_time} seconds')






