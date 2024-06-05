import numpy as np
import matplotlib.pyplot as plt

# Load the numpy files
training_loss = np.load('./train/train_loss.npy')
testing_loss = np.load('./val/val_loss.npy')
training_acc = np.load('./train/train_acc.npy')
val_acc = np.load('./val/val_acc.npy')

# print(val_acc)
# Plot training loss and testing loss for each epoch
plt.figure(figsize=(10, 5))
plt.plot(training_loss, label='Training Loss')
plt.plot(testing_loss, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('AGM - Training and Testing Loss per Epoch')
plt.legend()
plt.grid(True)
plt.savefig('loss_per_epoch.png')
plt.show()

# Plot training accuracy and validation accuracy for each epoch
plt.figure(figsize=(10, 5))
plt.plot(training_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('AGM - Training and Validation Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_per_epoch.png')
plt.show()