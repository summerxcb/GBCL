from config import *
from dataprocess import num_classes
class BERT_CNN_LSTM(nn.Module):
    def __init__(self, bert_model, lstm_hidden_size):
        super(BERT_CNN_LSTM, self).__init__()
        self.bert_model = bert_model
        self.convs = nn.ModuleList([nn.Conv2d(1, 200, (1, 768)) ,nn.Conv2d(1, 250, (2, 768)) ,nn.Conv2d(1, 300, (3, 768)) ])
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=750, hidden_size=lstm_hidden_size, num_layers=3, batch_first=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1, 128)
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        x = bert_output.unsqueeze(1)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(conv, conv.size(2)) for conv in x]
        x = torch.cat(x, dim=1)
        x=self.fc1(x)
        x=x.permute(0, 2, 1)
        lstm_output, _ = self.lstm(x)
        return lstm_output

class GCN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x
class CoAttention(nn.Module):
    def __init__(self, gcn_output_dim, lstm_hidden_dim, hidden_dim, num_classes, num_heads=1):
        super(CoAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.gcn_output_dim = gcn_output_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.W_lstm = nn.Linear(lstm_hidden_dim, hidden_dim * num_heads, bias=False)
        self.W_gcn = nn.Linear(gcn_output_dim, hidden_dim * num_heads, bias=False)
        self.W_bilinear = nn.Linear(hidden_dim * num_heads, num_heads, bias=False)
        self.layer_norm = nn.LayerNorm(gcn_output_dim + lstm_hidden_dim)
        self.classifier = nn.Linear(gcn_output_dim + lstm_hidden_dim, num_classes)
    def forward(self, gcn_output, lstm_output):
        batch_size, seq_len, _ = lstm_output.size()
        num_nodes = gcn_output.size(0)
        lstm_transformed = self.W_lstm(lstm_output)
        gcn_transformed = self.W_gcn(gcn_output)
        lstm_transformed = lstm_transformed.view(batch_size, seq_len, self.num_heads, self.hidden_dim)
        gcn_transformed = gcn_transformed.view(num_nodes, self.num_heads, self.hidden_dim)
        attention_matrix = torch.einsum('bsnh,nhd->bsnd', lstm_transformed,gcn_transformed)
        attention_matrix = self.W_bilinear(attention_matrix)
        attention_matrix = attention_matrix.view(batch_size, seq_len, num_nodes)
        attention_matrix = attention_matrix / math.sqrt(self.hidden_dim)
        gcn_attention_weights = F.softmax(attention_matrix, dim=2)
        lstm_attention_weights = F.softmax(attention_matrix.transpose(1, 2), dim=2)
        attended_gcn = torch.matmul(gcn_attention_weights, gcn_output)
        attended_lstm = torch.matmul(lstm_attention_weights, lstm_output)
        attended_lstm = attended_lstm.mean(dim=1)
        attended_lstm = attended_lstm.unsqueeze(1).expand(-1, attended_gcn.size(1),-1)
        combined = torch.cat([attended_gcn, attended_lstm],dim=-1)
        combined = self.layer_norm(combined)
        logits = self.classifier(combined.mean(dim=1))
        return logits
class CombinedModel(nn.Module):
    def __init__(self, bert_model, lstm_hidden_size, gcn_input_size, gcn_hidden_size,hidden_dim, num_classes):
        super(CombinedModel, self).__init__()
        self.bert_cnn_lstm = BERT_CNN_LSTM(bert_model, lstm_hidden_size)
        self.gcn = GCN(gcn_input_size, gcn_hidden_size)
        self.coattention = CoAttention(gcn_hidden_size,lstm_hidden_size, hidden_dim, num_classes,num_heads=1)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
    def forward(self, bert_input_ids, bert_attention_mask, gcn_input, gcn_edge_index,edge_weight):
        bert_output = self.bert_cnn_lstm(bert_input_ids, bert_attention_mask)
        gcn_output = self.gcn(gcn_input, gcn_edge_index,edge_weight)
        output = self.coattention(gcn_output,bert_output)
        return output
model = CombinedModel(bert_model, lstm_hidden_size, gcn_input_size, gcn_hidden_size,hidden_dim, num_classes)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
decay_rate = 0.95
decay_step = 2
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
