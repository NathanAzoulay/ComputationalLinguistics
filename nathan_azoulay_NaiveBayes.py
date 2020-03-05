#!/usr/bin/env python
# coding: utf-8

# In[21]:


# Nathan Azoulay
# March 5, 2020
# CS581 Professor Gawron

import nltk
from nltk.corpus import senseval
hard_data = [(i,i.senses[0]) for i in senseval.instances('hard.pos')]
len(hard_data)



senses = list(set(sense for (inst,sense) in hard_data))


def find_examples(data, sense, num_examples):
    res = []
    for (inst,s) in data:
        if s == sense:
           res.append(inst)
        if len(res) == num_examples:
           return res

def print_examples (ex_list):
    for ex in ex_list:
        for word in ex.context:
            print ('{0}_{1}'.format(word[0], word[1]))
        print()
        print()

from sklearn.utils import shuffle
new_hard_data = shuffle(hard_data,random_state=42)

train_ind = int(9 * round(len(hard_data)/10.))
train_data = new_hard_data[:train_ind]
test_data = new_hard_data[train_ind:]

#total amount of training data
len(train_data)

# Sort training data into three senses 
hard1_train = [s_inst for (s_inst, sense) in  train_data if sense == 'HARD1']
hard2_train = [s_inst for (s_inst, sense) in  train_data if sense == 'HARD2']
hard3_train = [s_inst for (s_inst, sense) in  train_data if sense == 'HARD3']

# Number of tokens of hard that bleongs with each sense
a1 = len(hard1_train)
a2 = len(hard2_train)
a3 = len(hard3_train)

#Question 1
#Total number of tokens of word hard
c = len(train_data)

p_hard1 = a1/c
p_hard2 = a2/c
p_hard3 = a3/c

# Made sure they add up to 1
p_total = p_hard1 + p_hard2 + p_hard3
print(p_hard1,p_hard2,p_hard3,p_total,'\n')


#Question 2 
si1 = hard1_train[0]
fd = nltk.FreqDist()
fd.update(si1.context)

hard1_fd = nltk.FreqDist()
for si in hard1_train:
    hard1_fd.update(si.context)
    
hard1_fd.N()

t0 = [('it', 'PRP'), ("'s", 'VBZ'), ('hard', 'JJ'), ('to', 'TO'), ('watch', 'VB'), ('.', '.')]

prob1 = (hard1_fd[('it', 'PRP')]/hard1_fd.N()) * (hard1_fd[("'s", 'VBZ')]/hard1_fd.N()) * (hard1_fd[('to', 'TO')]/hard1_fd.N()) * (hard1_fd[('watch', 'VB')]/hard1_fd.N()) * (hard1_fd[('.', '.')]/hard1_fd.N())



#Question 3 Hard2
si2 = hard2_train[0]
fd = nltk.FreqDist()
fd.update(si2.context)

hard2_fd = nltk.FreqDist()
for si in hard2_train:
    hard2_fd.update(si.context)
    
prob2 = (hard2_fd[('it', 'PRP')]/hard2_fd.N()) * (hard2_fd[("'s", 'VBZ')]/hard2_fd.N()) * (hard2_fd[('to', 'TO')]/hard2_fd.N()) * (hard2_fd[('watch', 'VB')]/hard2_fd.N()) * (hard2_fd[('.', '.')]/hard2_fd.N())



#Question 3 Hard3
si3 = hard3_train[0]
fd = nltk.FreqDist()
fd.update(si3.context)

hard3_fd = nltk.FreqDist()
for si in hard3_train:
    hard3_fd.update(si.context)

prob3 = (hard3_fd[('it', 'PRP')]/hard3_fd.N()) * (hard3_fd[("'s", 'VBZ')]/hard3_fd.N()) * (hard3_fd[('to', 'TO')]/hard3_fd.N()) * (hard3_fd[('watch', 'VB')]/hard3_fd.N()) * (hard3_fd[('.', '.')]/hard3_fd.N())
print(prob1,prob2,prob3,'\n')


# Question 4
total_V = set(list(hard1_fd.keys()) + list(hard2_fd.keys()) + list(hard3_fd.keys()))

#Smoothed hard1
sm_hard1_fd = nltk.FreqDist()
for word in total_V:
    sm_hard1_fd[word] = hard1_fd[word] + 1

ps1 = (sm_hard1_fd[('it', 'PRP')]/sm_hard1_fd.N()) * (sm_hard1_fd[("'s", 'VBZ')]/sm_hard1_fd.N()) * (sm_hard1_fd[('to', 'TO')]/sm_hard1_fd.N()) * (sm_hard1_fd[('watch', 'VB')]/sm_hard1_fd.N()) * (sm_hard1_fd[('.', '.')]/sm_hard1_fd.N())

#Smoothed hard2
sm_hard2_fd = nltk.FreqDist()
for word in total_V:
    sm_hard2_fd[word] = hard2_fd[word] + 1

ps2 = (sm_hard2_fd[('it', 'PRP')]/sm_hard2_fd.N()) * (sm_hard2_fd[("'s", 'VBZ')]/sm_hard2_fd.N()) * (sm_hard2_fd[('to', 'TO')]/sm_hard2_fd.N()) * (sm_hard2_fd[('watch', 'VB')]/sm_hard2_fd.N()) * (sm_hard2_fd[('.', '.')]/sm_hard2_fd.N())    
sm_hard3_fd = nltk.FreqDist()
for word in total_V:
    sm_hard3_fd[word] = hard3_fd[word] + 1
    
ps3 = (sm_hard1_fd[('it', 'PRP')]/sm_hard3_fd.N()) * (sm_hard3_fd[("'s", 'VBZ')]/sm_hard3_fd.N()) * (sm_hard3_fd[('to', 'TO')]/sm_hard3_fd.N()) * (sm_hard3_fd[('watch', 'VB')]/sm_hard3_fd.N()) * (sm_hard3_fd[('.', '.')]/sm_hard3_fd.N())    
print(ps1,ps2,ps3,'\n')


# Question 5
ans = prob1 / (prob1+prob2+prob3)
print(ans,'\n')


# ASSIGNMENT SOLUTIONS:
# 1 P(HARD1) = 0.7955
#   P(HARD2) = 0.1178
#   P(HARD3) = 0.0867
#   
# 2)Joint probability for Hard1 is 5.411251179818774e-11
# 3)Joint probability for Hard2 is 0.0
#   Joint probability for Hard3 is 1.1196360270456574e-12
#    
#   Naive Bayes chooses the HARD1 sense
# 
# 4)Smoothed probability hard1: 2.9610405090789285e-11
#   Smoothed probability hard2: 4.7326093497218936e-14
#   Smoothed probability hard3: 8.560509289089132e-13
#   
#   Classification decision made by smoothed model is hard1 sense.
#   
# 5)The probability using unsmoothed model, normalizing the probability is 0.9797285444957087
