from stanf import *
from config import *

class Graph:
    def __init__(self, sentence):
        self.sentence = sentence
        self.nlp_en = nlp_en
        self.nlp_zh = nlp_zh
        self.nlp_ar = nlp_ar
    def build_dependency_graph(self):
        lang, confidence = langid.classify(self.sentence)
        if lang == 'en':
            nlp = self.nlp_en
        elif lang == 'zh':
            nlp = self.nlp_zh
        elif lang == 'ar':
            nlp = self.nlp_ar
        else:
            nlp = self.nlp_ar
        doc = nlp(self.sentence)
        edges = []
        word_indices = {word.id: idx for idx, word in enumerate(doc.sentences[0].words)}
        for word in doc.sentences[0].words:
            if word.head != 0:  # Ignore root node
                head_idx = word_indices[word.head]
                dependent_idx = word_indices[word.id]
                edges.append((head_idx, dependent_idx))
        graph = nx.DiGraph(edges)
        return graph
    def compute_semantic_similarity(self):
        inputs = tokenizer(self.sentence, return_tensors="pt")
        inputs.to(device)
        bert_model.to(device)
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_size)
        num_tokens = embeddings.shape[0]
        sim_matrix = np.zeros((num_tokens, num_tokens))
        for i in range(num_tokens):
            for j in range(num_tokens):
                if i != j:
                    sim_matrix[i][j] = 1 - cosine(embeddings[i].cpu().detach().numpy(), embeddings[j].cpu().detach().numpy())
        return sim_matrix,embeddings
    def combine_dependency_and_similarity(self, dep_graph, sim_matrix, alpha=0.5):
        num_nodes = sim_matrix.shape[0]
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if dep_graph.has_edge(i, j) or dep_graph.has_edge(j, i):
                    adj_matrix[i][j] = 1
        adj_matrix = alpha * adj_matrix + (1 - alpha) * sim_matrix
        return torch.tensor(adj_matrix, dtype=torch.float)
    def sum(self):
        dep_graph = self.build_dependency_graph()
        sim_matrix,_ = self.compute_semantic_similarity()
        adj_matrix = self.combine_dependency_and_similarity(dep_graph, sim_matrix, alpha=0.5)
        adj_matrix_with_self_loops = adj_matrix + torch.eye(adj_matrix.size(0))
        deg_matrix = torch.diag(adj_matrix_with_self_loops.sum(dim=1))
        deg_matrix_inv_sqrt = torch.inverse(torch.sqrt(deg_matrix))
        normalized_adj_matrix = torch.mm(deg_matrix_inv_sqrt, torch.mm(adj_matrix_with_self_loops, deg_matrix_inv_sqrt))
        edge_index = normalized_adj_matrix.nonzero(as_tuple=False).t()
        edge_weight = normalized_adj_matrix[edge_index[0], edge_index[1]]
        return edge_index, edge_weight