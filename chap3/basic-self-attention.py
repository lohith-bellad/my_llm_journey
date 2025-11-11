import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

print("Inputs: \n", inputs)

query = inputs[1]

attn_scores_z2 = torch.empty(inputs.shape[0])
for ind, param in enumerate(inputs):
    attn_scores_z2[ind] = torch.dot(param, query)

print("Attn scores: ",attn_scores_z2)

attn_weights = torch.softmax(attn_scores_z2, dim=0)

print("Attn weights: ", attn_weights)

context_vec_2 = torch.zeros(query.shape)

for i, dim in enumerate(inputs):
    context_vec_2 += attn_weights[i] * dim

print("Context vector for 2nd word: ", context_vec_2)

all_attn_scores = inputs @ inputs.T
print("All attention scores: \n", all_attn_scores)

all_attn_weights = torch.softmax(all_attn_scores, dim=1)
print("All attention weights: \n", all_attn_weights)

all_context_vectors = all_attn_weights @ inputs
print("All context scores: \n", all_context_vectors)
