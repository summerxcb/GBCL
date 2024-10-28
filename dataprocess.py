from config import *
class LoadingData():
    def __init__(self, train_file_path, validation_file_path):
        self.train_file_path = train_file_path
        self.validation_file_path = validation_file_path
        category_id = 0
        self.cat_to_intent = {}
        self.intent_to_cat = {}
        for dirname, _, filenames in os.walk(train_file_path):
            for filename in filenames:
                file_path = os.path.join(dirname, filename)
                intent_id = filename.replace(".json", "")
                self.cat_to_intent[category_id] = intent_id
                self.intent_to_cat[intent_id] = category_id
                category_id += 1
        print(self.cat_to_intent)
        print(self.intent_to_cat)
        training_data = list()
        for dirname, _, filenames in os.walk(train_file_path):
            for filename in filenames:  #
                file_path = os.path.join(dirname, filename)
                intent_id = filename.replace(".json", "")
                training_data += self.make_data_for_intent_from_json(file_path, intent_id,self.intent_to_cat[intent_id])
        self.train_data_frame = pd.DataFrame(training_data,columns=['query', 'intent', 'category'])

        self.train_data_frame = self.train_data_frame.sample(frac=1)
        validation_data = list()
        for dirname, _, filenames in os.walk(validation_file_path):
            for filename in filenames:
                file_path = os.path.join(dirname, filename)
                intent_id = filename.replace(".json", "")
                validation_data += self.make_data_for_intent_from_json(file_path, intent_id,self.intent_to_cat[intent_id])
        self.validation_data_frame = pd.DataFrame(validation_data,columns=['query', 'intent', 'category'])
        self.validation_data_frame = self.validation_data_frame.sample(frac=1)  #
    def make_data_for_intent_from_json(self, json_file, intent_id, cat):
        json_d = json.load(open(json_file, encoding='utf-8', errors='ignore'))
        sent_list = list()
        for i in json_d:
            i = i["content"]
            # 处理缩写，将常见的英文缩写展开
            i = re.sub(r"\b(\w+)'(\w+)\b", r"\1 \2", i)
            i = re.sub(r"\b(\w+)'t\b", lambda match: match.group(1) + " not", i)
            i = re.sub(r"\b(\w+)'m\b", lambda match: match.group(1) + " am", i)
            # 删除链接地址
            i = re.sub(r'http\S+|www\S+', '', i)
            # 去除特殊字符和标点符号
            i = i.translate(str.maketrans('', '', string.punctuation))
            # 删除表情符号
            i = re.sub(r'[\U00010000-\U0010ffff]', '', i)
            # 删除无用的空格
            i = i.strip()
            i = re.sub(r'\s+', ' ', i)
            # 其他特定字符串替换
            i = i.replace("nan", "")
            i = i.replace("  ", "")
            i = i.replace(", , ", "")
            i = i.replace("laoshi", "")
            i = i.replace("hi", "")
            i = i.replace("hello", "")
            sent_list.append((i, intent_id, cat))
        return sent_list
ld = LoadingData(train_file_path, validation_file_path)
train_df = ld.train_data_frame
label_map, id2label = ld.intent_to_cat, ld.cat_to_intent
train_text, val_text, train_label, val_label = train_test_split(train_df['query'], train_df['category'],random_state=2018,test_size=0.2,stratify=train_df['category'])
train_texts,train_labels=train_df['query'], train_df['category']
test_df = ld.validation_data_frame
test_texts,test_labels=test_df['query'], test_df['category']
train_labels.reset_index(drop=True, inplace=True)
num_classes=len((np.unique(train_labels)))
class_labels = list(label_map.keys())
print(class_labels)
seq_len = [len(i.split()) for i in train_text]
pd.Series(seq_len).hist(bins=30)
max_seq_len = max(seq_len)
print(max_seq_len)
if max_seq_len>512:
    max_seq_len = 512
tokens_train = tokenizer.batch_encode_plus(train_text.tolist(),max_length=max_seq_len,pad_to_max_length=True,truncation=True,return_token_type_ids=False )
tokens_val = tokenizer.batch_encode_plus(val_text.tolist(),max_length=max_seq_len,pad_to_max_length=True,truncation=True,return_token_type_ids=False)
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_label.tolist())
print("train_y:", train_y)
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_label.tolist())
print("val_y:", val_y)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
tokens_test = tokenizer.batch_encode_plus(test_texts.tolist(),max_length=max_seq_len,pad_to_max_length=True,truncation=True,return_token_type_ids=False)
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())
for param in bert_model.parameters():
    param.requires_grad = False
