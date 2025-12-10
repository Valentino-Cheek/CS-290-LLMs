import math
# def softmax(v):
#     p = []
#     for i in range(len(v)):
#         p.append(math.exp(v[i]))
#     total = sum(p)
#     for i in range(len(v)):
#         p[i] /= total
#     return p

# #true values 1,0,2 indexes
# logit0 = [1.9, -2.1, -0.3, 0.4, -0.8]
# logit1 = [0.3, 2.5, -0.5, 0.0, -1.1]
# logit2 = [-0.6, 0.2, 1.4, -0.9, -0.2]

# softmax_logits = [softmax(logit0), softmax(logit1), softmax(logit2)]

# for i in range(len(softmax_logits)):
#     print(f"Softmax of logit {i}: {softmax_logits[i]}"
#           f"\n sum ={sum(softmax_logits[i])}\n")

# loss_values = []
# loss1 = -math.log(softmax_logits[0][1])  # true index 1 for logit0
# loss2 = -math.log(softmax_logits[1][0])  # true index 0 for logit1
# loss3 = -math.log(softmax_logits[2][2])  # true index 2 for logit2
# loss_values.append([loss1, loss2, loss3])

# for i in range(len(loss_values)):
#     print(f"Entropy of softmax logit {i}: {loss_values[i]}")

# avg_loss = (loss_values[0][0] + loss_values[0][1] + loss_values[0][2])/3
# print(f"Average Cross-Entropy Loss: {avg_loss}")

def softmaxsimple ( v ):
    exps = [ math.exp( i ) for i in v ]
    total = sum ( exps ) 
    return [ exps[i]/ total for i in range (len(exps)) ]

logits = []
logits.append( [1.9, -2.1, -0.3, 0.4, -0.8] )
logits.append( [0.3, 2.5, -0.5, 0.0, -1.1] )
logits.append ( [-0.6, 0.2, 1.4, -0.9, -0.2] )

targets = [1,0,2]

probs = [ softmaxsimple ( logits[i] ) for i in range ( len ( logits ) ) ]

def cross_entropy ( probs, targets ):
    return -math.log ( probs[targets] )

cross_entropies = [ cross_entropy ( probs[i], targets[i] ) for i in range ( len ( probs ) ) ]
print ( f"\nCross Entropies: { cross_entropies }" )

average_loss = sum ( cross_entropies ) / len ( cross_entropies )
print ( f"\nAverage Cross Entropy Loss: { average_loss } \n" )