# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from jaxtyping import Float, Int64

from .softtree_classification import SoftTreeClassifier

torch.manual_seed(404)

# %%
def STC_fit_NLLLoss_acc(
    X_train_tensor: Float[torch.Tensor, "_ _"],     # must be float32 flattened features for NLLLoss
    y_train_tensor: Int64[torch.Tensor, "_ _"],     # must be Long (int64) labels for NLLLoss
    X_test_tensor: Float[torch.Tensor, "_ _"],
    y_test_tensor: Int64[torch.Tensor, "_ _"],
    input_size,
    num_classes,
    tree_depth=5,
    beta=1.0, beta_epoch=1, beta_anneal=1.0,
    batch_size=1,
    num_epochs=1,
    learning_rate=0.01,
    lr_epoch=1, lr_decay=1.0,
    holdout_val=False,
    X_val_tensor: None | Float[torch.Tensor, "_ _"] = None,
    y_val_tensor: None | Int64[torch.Tensor, "_ _"] = None,
    print_every: None | int = None,
    lambda_l1=0.0,
    lambda_l2=0.0,
    lambda_groupl1=0.0,
):
    if holdout_val:
        assert X_val_tensor is not None
        assert y_val_tensor is not None

    # create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # create model
    model = SoftTreeClassifier(
        input_dim=input_size,
        output_dim=num_classes,
        depth=tree_depth,
        beta=beta,
        apply_batchNorm=False,
    )

    # loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if lr_decay < 1.0:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_epoch, gamma=lr_decay)

    # training loop
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0

        for X_batch, y_batch in train_loader:
            # forward
            train_outputs = model(X_batch)
            loss = criterion(train_outputs, y_batch)

            # regularization loss
            l1_reg, l2_reg, group_l1_reg = 0.0, 0.0, 0.0
            weights = model.inner_nodes.weight
            l1_reg = weights.abs().sum()
            l2_reg = weights.pow(2).sum()
            group_l1_reg = weights.pow(2).sum(dim=1).sqrt().sum()
            
            total_loss = loss + lambda_l1*l1_reg + lambda_l2*l2_reg + \
                lambda_groupl1*group_l1_reg

            # backward
            optimizer.zero_grad()    # clear previous gradients
            total_loss.backward()

            # update
            optimizer.step()
            
            # Track training error
            running_loss += loss.item()
            _, predicted = torch.max(train_outputs, 1)
            total_train += y_batch.size(0)
            correct_train += (predicted == y_batch).sum().item()

        # Calculate average training stats for this epoch
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train

        if print_every is not None and (epoch+1) % print_every == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        if holdout_val:
            # validation results
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

                _, val_predicted = torch.max(val_outputs, 1)
                total_val = y_test_tensor.size(0)
                correct_val = (val_predicted == y_val_tensor).sum().item()

                epoch_val_acc = correct_val / total_val

        # store results
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc)
        if holdout_val:
            val_loss_history.append(val_loss)
            val_acc_history.append(epoch_val_acc)
        
        # update scheduler
        if lr_decay < 1.0:
            scheduler.step()

        # update beta
        if beta_anneal > 1 and (epoch+1) % beta_epoch == 0:
            beta *= beta_anneal
            model.beta = beta

    # evaluation with test set
    model.eval()
    with torch.no_grad():
        prob_test = model(X_test_tensor)
        _, y_test_pred = torch.max(prob_test.data, 1)
        accuracy = (y_test_pred == y_test_tensor).sum().item() / len(y_test_tensor)
        if print_every is not None:
            print(f'Test Accuracy: {accuracy:.4f}')
    
    return model, accuracy, train_loss_history, train_acc_history, val_loss_history, val_acc_history

# %%
if __name__ == '__main__':
    # hyperparameters
    input_size = 2
    num_classes = 4

    tree_depth = 3
    # tree_depth = 2
    beta, beta_epoch, beta_anneal = 1.0, 1, 1.0

    batch_size = 32
    # batch_size = 1
    num_epochs = 100
    learning_rate = 0.01
    lr_epoch, lr_decay = 1, 1.0
    lmd_l1, lmd_l2, lmd_gl1 = 0.01, 0.0, 0.0

    # load data
    # X = features (coordinates), y = labels (0, 1, 2, 3)
    imported_data = np.load('data/make_gaussian_1000_seed42.npz')
    X_train, y_train = imported_data['X_train'], imported_data['y_train']
    X_val, y_val = imported_data['X_val'], imported_data['y_val']
    X_test, y_test = imported_data['X_test'], imported_data['y_test']

    # prepare pytorch data
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # implement training
    STC_model, test_accuracy, train_loss_history, train_acc_history, \
        val_loss_history, val_acc_history = STC_fit_NLLLoss_acc(
            X_train_tensor, y_train_tensor,
            X_test_tensor, y_test_tensor,
            input_size, num_classes,
            tree_depth=tree_depth,
            beta=beta, beta_epoch=beta_epoch, beta_anneal=beta_anneal,
            batch_size=batch_size, num_epochs=num_epochs,
            learning_rate=learning_rate,
            lr_epoch=lr_epoch, lr_decay=lr_decay,
            print_every=10,
            holdout_val=True, X_val_tensor=X_val_tensor, y_val_tensor=y_val_tensor,
            lambda_l1=lmd_l1,
            lambda_l2=lmd_l2,
            lambda_groupl1=lmd_gl1
    )

    # save model
    torch.save(STC_model.state_dict(), 'models/STC_make_gaussian_1000_seed42.pt')
    np.savez(
        'models/STC_make_gaussian_1000_seed42.npz',
        input_size=input_size,
        num_classes=num_classes,
        tree_depth=tree_depth,
        beta=beta, beta_epoch=beta_epoch, beta_anneal=beta_anneal,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        lr_epoch=lr_epoch, lr_decay=lr_decay,
        lmd_l1=lmd_l1, lmd_l2=lmd_l2, lmd_gl1=lmd_gl1,
        test_accuracy=test_accuracy,
        train_loss_history=train_loss_history,
        train_acc_history=train_acc_history,
        val_loss_history=val_loss_history,
        val_acc_history=val_acc_history
    )

    # append results to csvexcel
    results_file = 'results/make_gaussian.csv'
    header = [
        'tree_depth', 'beta', 'beta_epoch', 'beta_anneal',
        'batch_size', 'num_epochs',
        'learning_rate', 'lr_epoch', 'lr_decay',
        'lmd_l1', 'lmd_l2', 'lmd_gl1',
        'train_STC', 'val_STC', 'test_STC'
    ]
    data = [
        tree_depth, beta, beta_epoch, beta_anneal,
        batch_size, num_epochs,
        learning_rate, lr_epoch, lr_decay,
        lmd_l1, lmd_l2, lmd_gl1,
        train_acc_history[-1], val_acc_history[-1], test_accuracy
    ]

    file_exists = os.path.exists(results_file)
    with open(results_file, mode='a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(header)
        writer.writerow(data)    

# %%
# test loaded model (must run previous cell first to have test set)
if __name__ == '__main__':
    model_hypers = np.load('models/STC_make_gaussian_1000_seed42.npz')
    input_size = model_hypers['input_size'].item()
    num_classes = model_hypers['num_classes'].item()
    tree_depth = model_hypers['tree_depth'].item()
    beta = model_hypers['beta'].item()

    loaded_STC_model = SoftTreeClassifier(
        input_dim=input_size,
        output_dim=num_classes,
        depth=tree_depth,
        beta=beta,
        apply_batchNorm=False,
    )

    loaded_STC_model.load_state_dict(torch.load('models/STC_make_gaussian_1000_seed42.pt'))
    loaded_STC_model.eval()

    with torch.no_grad():
        prob_test = loaded_STC_model(X_test_tensor)
        _, y_test_pred = torch.max(prob_test.data, 1)
        accuracy = (y_test_pred == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f'Test Accuracy (loaded model): {accuracy:.4f}')

# %%
# visualize learning curves
import matplotlib.pyplot as plt

if __name__ == '__main__':

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    ax1.plot(train_loss_history, label='Train Loss')
    ax1.plot(val_loss_history, label='Validation Loss', color='orange')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Accuracy
    ax2.plot(train_acc_history, label='Train Accuracy')
    ax2.plot(val_acc_history, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.show()

# %%