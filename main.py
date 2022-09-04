from transformers import AutoTokenizer, AutoModel
import torch

model_name = 'sentence-transformers/bert-base-nli-mean-tokens'

#Create last hidden state tensor.

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

#Data to find similarity

sentences = []

#Tokenize all the sentences
tokens = {'input_ids' : [], 'attention_mask' : []}

for sentence in sentences:
    new_tokens = tokenizer.encode_plus(sentence, max_length = 128, truncation = True, padding = 'max_length',return_tensors = 'pt') 
    tokens['input_ids'].append(new_tokens['input_ids'][0])
    tokens['attention_mask'].append(new_tokens['attention_mask'][0])

#Stacking all the torch tensors as a single list

tokens['input_ids'] = torch.stack(tokens['input_ids'])
tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
print(tokens['input_ids'].shape)

outputs = model(**tokens)
print(outputs.keys())

#Obtainig model embeddings i.e Contextual embeddings
embeddings = outputs.last_hidden_state
print(embeddings.shape)  #Usually [6,128,768]

#Creating sentence vectors
attention = tokens['attention_mask']
print(attention.shape) #Shape --> [6,128]

#Note: we need to add another dimenstion of "768" to the attention and multiply it with the o/p vectors to remove the embeddings which are padded.
mask = attention.unsqueeze(-1).expand(embeddings.shape).float()
mask_embeddings = embeddings * mask

print(mask_embeddings.shape) #We get the shape [6,128,768]

#Note: we need to reduce 128 --> 1. We do it using Mean Pooling.

summed = torch.sum(mask_embeddings,1) #Dimension : 1
print(summed.shape)

#Note : Count only where we pay attention, i.e : 1. We ignore 0's 
counts = torch.clamp(mask.sum(1),min = 1e-9) #Dimension : 1 , min : Stops by performing divide by 0 error.

mean_pooled = summed / counts
print(mean_pooled.shape)

#Performing Cosine Similarity.

from sklear.metrics.pairwise import cosine_similarity

#Note: We need to convert it into Numpy array before we find the cosine similarity.
mean_pooled = mean_pooled.detach().numpy()

sim = cosine_similarity(
    [mean_pooled[0]],
    mean_pooled[1:]
)

print(sim)

 

