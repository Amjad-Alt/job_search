from datasets import load_dataset,Dataset,DatasetDict
from transformers import DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import pandas as pd
#%%
# (Cloud) Move to the file directory
import os
path = r'/home/ubuntu/NLP_local/4. Transformer_Class_ex_9_10' #
os.chdir(path)
print(os.getcwd())
#%%
###############################################
# 1. Load input data (Kaggle) and Preprocessing
###############################################
# News Headlines Dataset For Sarcas
# !mkdir ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d rmisra/news-headlines-dataset-for-sarcasm-detection
#%%
# Load data to use in Transformer (load_dataset())
data=load_dataset("json",data_files="./Sarcasm_Headlines_Dataset_v2.json")
#%%
data #28619,
#%%
# Explore data (using Pandas)
temp = pd.read_json("./Sarcasm_Headlines_Dataset_v2.json", lines=True)
temp.head()
temp.is_sarcastic.value_counts() #2 categories

#%%
# preprocessing data (format: transformer - pandas - transformer)
data=data.rename_column("is_sarcastic","label")
data=data.remove_columns(['article_link'])

data.set_format('pandas')
data=data['train'][:]

data.drop_duplicates(subset=['headline'],inplace=True)
data=data.reset_index()[['headline','label']]
data=Dataset.from_pandas(data)
#%%
data  #28503
#%%
# 80% train, 20% (10% test + 10% validation)
train_testvalid = data.train_test_split(test_size=0.2,seed=15)  #train, test(20%: disgard)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5,seed=15)  #train:valid, test: test

#%%
# gather everyone if you want to have a single DatasetDict
data = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})
data
#%%
data['train'][0]
#%%
###############################################
# [Train with Customized model]
#    A. Inpt data Preprocessing (Use origin Tokenizer)
#    B. ★ Build CustomModel
#       - Extract BODY
#       - Add New Head
#         : Add Dropout Layer, Extract embedded output, Dense layer for 2 categories
#    C. Train with Customized module
###############################################

#%%
checkpoint = "cardiffnlp/twitter-roberta-base-emotion"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.model_max_len=512
#%%
def tokenize(batch):
  return tokenizer(batch["headline"], truncation=True,max_length=512)

tokenized_dataset = data.map(tokenize, batched=True)
tokenized_dataset
#%%
tokenized_dataset['train'][0]
#%%
tokenized_dataset.set_format("torch",columns=["input_ids", "attention_mask", "label"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#Option: used in BERT(language modeling and applies random masking)
#data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
#%%
class CustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super(CustomModel, self).__init__()
        self.num_labels = num_labels

        # Load Model with given checkpoint and extract its body
        self.model = model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint,
                                                                                                     output_attentions=True,
                                                                                                        output_hidden_states=True))
        #The Dropout layer: used to prevent overfitting
        self.dropout = nn.Dropout(0.1)  # 10%: set to zero when training
        self.classifier = nn.Linear(768, num_labels)  # load and initialize weights
        # 768: Bert Base (# of hidden layers, w 12 attention layers)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        # CLS token
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Add custom layers
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state

        logits = self.classifier(sequence_output[:, 0, :].view(-1, 768))  # calculate losses

        loss = None
        #- Loss Function
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Option: BCELoss(): Binary classification
            #loss_fct = nn.BCELoss()

            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)

#%%
# GPU(device), Define Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CustomModel(checkpoint=checkpoint,num_labels=2).to(device)
#%%
#  (Define) Dataloader
batch_size = 32
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_dataset["valid"], batch_size=batch_size, collate_fn=data_collator
)
#%%
# Set Hyperparameters,
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr=5e-5
## Optimizer
from transformers import AdamW,get_scheduler
optimizer = AdamW(model.parameters(), lr=lr)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

#%%
# Define Evaluate metric (classification)
from datasets import load_metric
metric = load_metric("f1")

#%%
# (Define) Training(w train_dataloader)/Evaluate(w test_dataloader)/Training loop (Epoch)
from tqdm.auto import tqdm

progress_bar_train = tqdm(range(num_training_steps))
progress_bar_eval = tqdm(range(num_epochs * len(eval_dataloader)))

for epoch in range(num_epochs):
    model.train()  #Call train function
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch) # train
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar_train.update(1)

    model.eval() #Call eval function
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        progress_bar_eval.update(1)

    print(metric.compute())  # check performance on validation set

#%%
# Check final performance on test set
model.eval()

test_dataloader = DataLoader(
    tokenized_dataset["test"], batch_size=32, collate_fn=data_collator
)

for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()

#%%M
model_path = "./Model_Nammin_Woo.pt"  # specify the path to save the model
torch.save(model.state_dict(), model_path)
#%%
