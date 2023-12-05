import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from torch.optim import Adam


# Define the Siamese Bert Model
class SiameseBert(nn.Module):
    def __init__(self):
        super(SiameseBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        # Pass the first input through BERT
        outputs_1 = self.bert(input_ids=input_ids_1, attention_mask=attention_mask_1)
        # Only use the pooled output
        pooled_output_1 = outputs_1.pooler_output

        # Pass the second input through BERT
        outputs_2 = self.bert(input_ids=input_ids_2, attention_mask=attention_mask_2)
        # Only use the pooled output
        pooled_output_2 = outputs_2.pooler_output

        # Return the pooled outputs for both inputs
        return pooled_output_1, pooled_output_2


# Function to calculate the contrastive loss
def contrastive_loss(output_1, output_2, label, margin=1.0):
    euclidean_distance = nn.functional.pairwise_distance(output_1, output_2)
    loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive


# Example usage
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = SiameseBert()

# Simulated inputs
input_text_1 = "Example resume text."
input_text_2 = "Example job description."

# Tokenize inputs
encoded_input_1 = tokenizer(input_text_1, return_tensors='pt', padding=True, truncation=True, max_length=128)
encoded_input_2 = tokenizer(input_text_2, return_tensors='pt', padding=True, truncation=True, max_length=128)

# Simulated label (1 for similar, 0 for dissimilar)
label = torch.tensor([1], dtype=torch.float)

# Forward pass, unpacking the tokenized inputs explicitly
output_1, output_2 = model(
    input_ids_1=encoded_input_1['input_ids'],
    attention_mask_1=encoded_input_1['attention_mask'],
    input_ids_2=encoded_input_2['input_ids'],
    attention_mask_2=encoded_input_2['attention_mask']
)

# Calculate loss
loss = contrastive_loss(output_1, output_2, label)

# Backward pass and optimization
optimizer = Adam(model.parameters())
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
