import sys
sys.path.append("../")

from fastNLP import DataSet

data_path = "../tutorials/sample_data/tutorial_sample_dataset.csv"
ds = DataSet.read_csv(data_path, headers=('raw_sentence', 'label'), sep='\t') 




# 将所有数字转为小写
ds.apply(lambda x: x['raw_sentence'].lower(), new_field_name='raw_sentence')
# label转int
ds.apply(lambda x: int(x['label']), new_field_name='label_seq', is_target=True)

def split_sent(ins):
    return ins['raw_sentence'].split()
ds.apply(split_sent, new_field_name='words', is_input=True)




# 分割训练集/验证集
train_data, dev_data = ds.split(0.3)
print("Train size: ", len(train_data))
print("Test size: ", len(dev_data))




from fastNLP import Vocabulary
vocab = Vocabulary(min_freq=2)
train_data.apply(lambda x: [vocab.add(word) for word in x['words']])

# index句子, Vocabulary.to_index(word)
train_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='word_seq', is_input=True)
dev_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='word_seq', is_input=True)

train_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='translated_seq', is_input=True)
dev_data.apply(lambda x: [vocab.to_index(word) for word in x['words']], new_field_name='translated_seq', is_input=True)

print(train_data[3])

from fastNLP.models import CNNText
from fastNLP.models import Transformer
# model = CNNText(embed_num=len(vocab), embed_dim=50, num_classes=5, padding=2, dropout=0.1)
model = Transformer(embed_num=len(vocab), embed_dim=50, num_classes=5, padding=2, dropout=0.1)

from fastNLP import Trainer, CrossEntropyLoss, AccuracyMetric
trainer = Trainer(model=model, 
                  train_data=train_data, 
                  dev_data=dev_data,
                  loss=CrossEntropyLoss(),
                  metrics=AccuracyMetric()
                  )
trainer.train()
print('Train finished!')



