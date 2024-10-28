from config import *
from dataprocess import *
from graph import Graph
from Model import model,criterion,optimizer
class Train():
    def train(self):
        model.train()
        total_loss = 0
        total_accuracy = 0
        total_preds = []
        total_labels = []
        for step, batch in enumerate(train_dataloader):
            if step % 20000 == 0 and step != 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
            batch = [r.to(device) for r in batch]
            sent_id, mask, labels = batch
            input_ids_list = sent_id.tolist()
            sentence_list = tokenizer.batch_decode(input_ids_list, skip_special_tokens=True)

            for sentence, single_sent_id, single_mask, single_label in zip(sentence_list, sent_id, mask, labels):
                a = Graph(sentence)
                edge_index, edge_weight = a.sum()
                _, embeddings = a.compute_semantic_similarity()
                embeddings = embeddings.to(device)
                edge_index = edge_index.to(device)
                edge_weight = edge_weight.to(device)
                data = Data(x=embeddings, edge_index=edge_index, edge_weight=edge_weight)
                data = data.to(device)
                model.zero_grad()
                preds = model(single_sent_id.unsqueeze(0), single_mask.unsqueeze(0), data.x, data.edge_index,data.edge_weight)
                loss = criterion(preds, single_label.unsqueeze(0))
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                preds = preds.detach().cpu().numpy()
                preds = np.argmax(preds, axis=1)
                total_preds += list(preds)
                total_labels += [single_label.item()]
        avg_loss = (total_loss / len(train_dataloader))/len(epochs)
        print('avg_loss', avg_loss)
        f1 = f1_score(total_labels, total_preds, average='weighted')
        accuracy = accuracy_score(total_labels, total_preds)
        precision = precision_score(total_labels, total_preds, average='weighted')
        recall = recall_score(total_labels, total_preds, average='weighted')
        return avg_loss, f1, accuracy, precision, recall
    def evaluate(self):
        print("\nevaluations...")
        model.eval()
        total_loss, total_accuracy = 0, 0
        total_preds = []
        total_labels = []
        for step, batch in enumerate(val_dataloader):
            if step % 5000 == 0 and step != 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
            batch = [t.to(device) for t in batch]
            sent_id, mask, labels = batch
            with torch.no_grad():
                input_ids_list = sent_id.tolist()
                sentence_list = tokenizer.batch_decode(input_ids_list, skip_special_tokens=True)
                for sentence, single_sent_id, single_mask, single_label in zip(sentence_list, sent_id, mask, labels):
                    a = Graph(sentence)
                    edge_index, edge_weight = a.sum()
                    _, embeddings = a.compute_semantic_similarity()
                    embeddings = embeddings.to(device)
                    edge_index = edge_index.to(device)
                    edge_weight = edge_weight.to(device)
                    data = Data(x=embeddings, edge_index=edge_index, edge_weight=edge_weight)
                    data = data.to(device)
                    preds = model(single_sent_id.unsqueeze(0), single_mask.unsqueeze(0), data.x, data.edge_index,data.edge_weight)
                    loss = criterion(preds, single_label.unsqueeze(0))
                    total_loss += loss.item()
                    preds = preds.detach().cpu().numpy()
                    preds = np.argmax(preds, axis=1)
                    total_preds += list(preds)
                    total_labels += [single_label.item()]
        avg_loss = (total_loss / len(val_dataloader))/epochs
        f1 = f1_score(total_labels, total_preds, average='weighted')
        accuracy = accuracy_score(total_labels, total_preds)
        precision = precision_score(total_labels, total_preds, average='weighted')
        recall = recall_score(total_labels, total_preds, average='weighted')
        return avg_loss, f1, accuracy, precision, recall
trainer = Train()
