from config import *
from dataprocess import *
from graph import *
from Model import criterion
path = 'path/to/your_best_model'
test_df = ld.validation_data_frame
checkpoint = torch.load(path, map_location=device)
model = checkpoint.get("model")
model.to(device)
toloss=0
all_preds = []
for i in range(len(test_seq)):
    with torch.no_grad():
        decoded_texts = tokenizer.decode(test_seq[i].tolist(), skip_special_tokens=True)
        a = Graph(decoded_texts)
        edge_index, edge_weight = a.sum()
        _, embeddings = a.compute_semantic_similarity()
        data = Data(x=embeddings.to(device),edge_index=edge_index.to(device),edge_weight=edge_weight.to(device))
        seq_device = test_seq[i].unsqueeze(0).to(device)
        mask_device = test_mask[i].unsqueeze(0).to(device)
        labels=test_y[i].unsqueeze(0).to(device)
        pred = model(seq_device, mask_device, data.x, data.edge_index, data.edge_weight)
        loss = criterion(pred,labels )
        pred = pred.detach().cpu().numpy()
        toloss+=loss.item()
        pred_label = np.argmax(pred, axis=1)
        all_preds.append(pred_label)
avg_loss = toloss / len(test_seq)
all_preds = np.concatenate(all_preds)
print(classification_report(test_y, all_preds))
report = classification_report(test_y, all_preds, output_dict=True)
print(f"Average loss: {avg_loss:.4f}")

formatted_report = {}
for key, metrics in report.items():
    if isinstance(metrics, dict):
        formatted_metrics = {metric: round(value, 4) for metric, value in metrics.items()}
        formatted_report[key] = formatted_metrics
    else:
        formatted_report[key] = round(metrics, 4)
for key, metrics in formatted_report.items():
    if isinstance(metrics, dict):
        print(f"{key}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    else:
        print(f"{key}: {metrics}")
