import torch
import torch.nn as nn

def accuracy(logits, targets):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds == targets).float().mean().item()


def TrainingAlgorithm(model, train_loader, val_loader, num_epochs, patience=10, device="cpu"):
    """
    Train the model on train_loader and evaluate on eval_loader.
    Return lists train_losses: avarage trainingloss per epoch, and 
    eval_losses: avarage validationloss per epoch
    """
    model.to(device)

    train_losses = []
    eval_losses = []
    eval_accuracies = []

    # Loss-function og optimizer
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(num_epochs):

        # -------- TRAINING --------
        model.train()
        total_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            # Forward pass
            logits = model(X_batch)
            loss = loss_func(logits, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------- VALIDATION --------
        model.eval()
        total_eval_loss = 0.0
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device).float()
                y_val = y_val.to(device).float()

                logits_val = model(X_val)
                loss_val = loss_func(logits_val, y_val)

                total_eval_loss += loss_val.item()

                all_logits.append(logits_val)
                all_targets.append(y_val)

        avg_eval_loss = total_eval_loss / len(val_loader)
        eval_losses.append(avg_eval_loss)

        logits_cat = torch.cat(all_logits)
        targets_cat = torch.cat(all_targets)
        val_acc = accuracy(logits_cat, targets_cat)
        eval_accuracies.append(val_acc)

        # -------- EARLY STOPPING --------
        if avg_eval_loss < best_val_loss:
            best_val_loss = avg_eval_loss
            best_state_dict = model.state_dict()  
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"- train loss: {avg_train_loss:.4f} "
              f"- val loss: {avg_eval_loss:.4f} "
              f"- val acc: {val_acc:.4f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # -------- SAVE WEIGHTS --------
    if best_state_dict is not None:
        torch.save(best_state_dict, "model_weights.pth")
    else:
        torch.save(model.state_dict(), "model_weights.pth")
        print("Something wrong")

    train_losses = train_losses[:best_epoch+1]
    eval_losses = eval_losses[:best_epoch+1]
    eval_accuracies = eval_accuracies[:best_epoch+1]

    return train_losses, eval_losses, eval_accuracies