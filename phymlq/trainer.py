import torch
import tqdm
import wandb

from phymlq.hyperparams import DEVICE


def train(model, criterion, optimizer, train_dataloader, val_dataloader, epochs=10, use_wandb=False):
    batch_losses, batch_accuracy = [], []
    val_batch_losses, val_batch_accuracy = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, total_samples, total_correct = 0, 0, 0
        train_iterator = tqdm.tqdm(train_dataloader)
        train_iterator.set_description("Training Epoch %d" % (epoch + 1))
        for batch in train_iterator:
            optimizer.zero_grad()

            batch = batch.to(DEVICE)
            y = torch.tensor(batch.y).to(DEVICE)
            out = model(batch)

            loss = criterion(out, y)
            total_loss += loss.item()
            total_samples += y.shape[0]
            # noinspection PyUnresolvedReferences
            total_correct += (torch.max(out, 1)[1] == y).float().sum().item()
            loss.backward()
            optimizer.step()

            train_iterator.set_postfix(
                loss=total_loss / total_samples,
                accuracy=total_correct / total_samples)

        batch_losses.append(total_loss / total_samples)
        batch_accuracy.append(total_correct / total_samples)
        if use_wandb:
            wandb.log({'Training Accuracy': batch_accuracy[-1],
                       'Training Loss': batch_losses[-1]})

        total_loss, total_samples, total_correct = 0, 0, 0
        with torch.no_grad():
            model.eval()
            val_iterator = tqdm.tqdm(val_dataloader)
            val_iterator.set_description("Validation Epoch %d" % (epoch + 1))
            for batch in val_iterator:
                batch = batch.to(DEVICE)
                y = torch.tensor(batch.y).to(DEVICE)
                out = model(batch)

                loss = criterion(out, y)
                total_loss += loss.item()
                total_samples += y.shape[0]
                # noinspection PyUnresolvedReferences
                total_correct += (torch.max(out, 1)[1] == y).float().sum().item()

                val_iterator.set_postfix(
                    loss=total_loss / total_samples,
                    accuracy=total_correct / total_samples)

            val_batch_losses.append(total_loss / total_samples)
            val_batch_accuracy.append(total_correct / total_samples)
        if use_wandb:
            wandb.log({'Validation Accuracy': val_batch_accuracy[-1],
                       'Validation Loss': val_batch_losses[-1]})
