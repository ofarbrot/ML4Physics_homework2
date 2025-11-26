import torch
import torch.nn as nn

def TrainingAlgorithm(model, train_loader, val_loader, num_epochs, patience=10, device="cpu"):
    """
    Train the model on train_loader and evaluate on val_loader for a REGRESSION task.
    Returns: train_losses, eval_losses
    """
    model.to(device)

    train_losses = []
    eval_losses = []

    # Regression loss
    loss_func = nn.MSELoss()
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

            preds = model(X_batch)               # predicted alpha
            preds = preds.view_as(y_batch)       # ensure same shape

            loss = loss_func(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------- VALIDATION --------
        model.eval()
        total_eval_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device).float()
                y_val = y_val.to(device).float()

                preds_val = model(X_val)
                preds_val = preds_val.view_as(y_val)

                loss_val = loss_func(preds_val, y_val)

                total_eval_loss += loss_val.item()
                all_preds.append(preds_val)
                all_targets.append(y_val)

        avg_eval_loss = total_eval_loss / len(val_loader)
        eval_losses.append(avg_eval_loss)

        # (Optional) extra metric: MAE
        preds_cat = torch.cat(all_preds).view(-1)
        targets_cat = torch.cat(all_targets).view(-1)
        val_mae = torch.mean(torch.abs(preds_cat - targets_cat)).item()

        # -------- EARLY STOPPING --------
        if avg_eval_loss < best_val_loss:
            best_val_loss = avg_eval_loss
            best_state_dict = model.state_dict()
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- train MSE: {avg_train_loss:.4f} "
            f"- val MSE: {avg_eval_loss:.4f} "
            f"- val MAE: {val_mae:.4f}"
        )

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # -------- SAVE WEIGHTS --------
    if best_state_dict is not None:
        torch.save(best_state_dict, "model_weights.pth")
    else:
        torch.save(model.state_dict(), "model_weights.pth")
        print("Warning: no best_state_dict, saved last model instead.")

    train_losses = train_losses[:best_epoch+1]
    eval_losses = eval_losses[:best_epoch+1]

    return train_losses, eval_losses