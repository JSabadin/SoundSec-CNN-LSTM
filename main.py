# This file is derived from work originally created by Hbbbbbby (https://github.com/Hbbbbbby/EmotionRecognition_2Dcnn-lstm).
# Original License: BSD 3-Clause License (https://github.com/Hbbbbbby/EmotionRecognition_2Dcnn-lstm/blob/main/LICENSE).
# Changes were made by adding stopping criteria, plotting confusion matrices, and using a different dataset.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cnn2d  # Ensure this module is adapted for PyTorch
import dataload  # Ensure this module is adapted for PyTorch

# Configuration for using GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, optimizer, criterion, train_loader, validation_loader, device, epochs=10, patience=20):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    # Lists for storing metrics and predictions
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    all_preds = []
    all_labels = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect predictions and actual labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(validation_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Check if both train and validation accuracies are 100%
        if train_accuracy == 100.0 and val_accuracy == 100.0:
            print("Both training and validation accuracies reached 100%. Stopping training.")
            torch.save(model.state_dict(), 'best_model.pth')
            break

        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter > patience:
            print("Early stopping due to lack of improvement in validation loss")
            break
        
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

def test(model, test_loader, device):
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test data: {accuracy}%')

    # Labels for the confusion matrix
    sound_labels = ['siren', 'bark', 'shoot', 'scream', 'glass', 'alarm', 'explosion']

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Plot the confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sound_labels, yticklabels=sound_labels)
    plt.title('Confusion Matrix - Testing')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == '__main__':
    EmoDB_file_path = 'C:\\Users\\jerne\\OneDrive\\Desktop\\FAKS\\MAG2\\Govorne_tehnologije\\diploma_extended\\wav'

    # Load data
    train_loader, validation_loader, test_loader = dataload.load_data(EmoDB_file_path)

    for inputs, targets in train_loader:
        print(f"Input batch shape: {inputs.shape}")
        print(f"Target batch shape: {targets.shape}")
        break

    # Create model and optimizer
    input_shape = (1, 128, 251)  # Adjust as needed
    num_classes = 7  # Adjust as needed
    model, optimizer = cnn2d.create_model2d(input_shape, num_classes)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train(model, optimizer, criterion, train_loader, validation_loader, device)

    # Test the model
    test(model, test_loader, device)
