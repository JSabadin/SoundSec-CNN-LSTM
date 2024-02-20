import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import network  
import dataload  

# Configuration for using GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, optimizer, criterion, train_loader, validation_loader, device, epochs=30, patience=20):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    # Lists for storing metrics and predictions
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

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

    # Calculate precision, recall, and F1 score
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    # Labels for the confusion matrix
    emotion_labels = ['anger', 'Boredom', 'disgust', 'fear', 'happiness', 'sadness', 'neutral']

    # Plotting the confusion matrix for testing
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    sns.set(font_scale=0.7)  # Adjust the font scale as needed
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title('Confusion Matrix - Testing')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)  # Rotate x labels for better visibility
    plt.yticks(rotation=0)  # Keep y labels horizontal
    plt.show()

if __name__ == '__main__':
    dataset_path = 'C:\\Users\\jerne\\Desktop\\FAKS\\MAG2\\Govorne_tehnologije\\diploma_extended\\wav'
    
    # Load data
    train_loader, validation_loader, test_loader = dataload.load_data(dataset_path)

    for inputs, targets in train_loader:
        print(f"Input batch shape: {inputs.shape}")
        print(f"Target batch shape: {targets.shape}")
        break

    # Create model and optimizer
    model, optimizer = network.create_model(num_features=39, num_time_frames=251, num_classes=7)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train(model, optimizer, criterion, train_loader, validation_loader, device)

    # Test the model
    test(model, test_loader, device)
