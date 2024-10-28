
from config import *
from Train import trainer
from dataprocess import label_map, id2label
from Model import model, optimizer, decay_step, scheduler

def save_checkpoint(filename, epoch, model, optimizer, label_map, id2label):
    state = {
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,
        'label_map': label_map,
        'id_map': id2label}
    torch.save(state, filename)

def main():
    best_valid_loss = float('inf')
    max_f1_valid = float('-inf')
    max_accuracy = float('-inf')
    max_precision = float('-inf')
    max_recall = float('-inf')
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        train_loss, f1_train, accuracy_train, precision_train, recall_train = trainer.train()
        valid_loss, f1_valid, accuracy_valid, precision_valid, recall_valid = trainer.evaluate()

        if f1_valid > max_f1_valid:
            max_f1_valid = f1_valid
        if accuracy_valid > max_accuracy:
            max_accuracy = accuracy_valid
        if precision_valid > max_precision:
            max_precision = precision_valid
        if recall_valid > max_recall:
            max_recall = recall_valid
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            file_name = 'best_model.pt'
            save_checkpoint(file_name, epoch, model, optimizer, label_map, id2label)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\ntrain_loss: {train_loss:.3f}')
        print(f'val_loss: {valid_loss:.3f}')
        print(f'\ntrain F1: {f1_train:.3f}')
        print(f'val F1: {f1_valid:.3f}')
        print(f'train acc: {accuracy_train:.3f}')
        print(f'train pre: {precision_train:.3f}')
        print(f'train recall: {recall_train:.3f}')
        print(f'val acc: {accuracy_valid:.3f}')
        print(f'val pre: {precision_valid:.3f}')
        print(f'val recall: {recall_valid:.3f}')

        if (epoch + 1) % decay_step == 0:
            scheduler.step()

    print("max f1_valid:", max_f1_valid)
    print("max accuracy:", max_accuracy)
    print("max precision:", max_precision)
    print("max recall:", max_recall)
    print('\n')

if __name__ == "__main__":
    main()
