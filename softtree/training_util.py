# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from jaxtyping import Float, Int64

from .softtree_classification import SoftTreeClassifier


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