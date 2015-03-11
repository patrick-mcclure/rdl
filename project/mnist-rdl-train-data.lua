require 'torch'
require 'nn'
require 'optim'
require 'dataset-mnist'
require 'pl'
require 'paths'

classes = {'1','2','3','4','5','6','7','8','9','10'}

n = 10

count = torch.Tensor(#classes)

trainData = mnist.loadTrainSet(60000, {32, 32})

r = torch.randperm(trainData:size(1))

trainIndex = torch.Tensor(#classes,n)

i = 1
flg = true
j = 1
while flg do
  local _,target = trainData[r[i]][2]:clone():max(1)
  local t = target:squeeze()

  if count[t] < n then
    count[t] = count[t]+1
    trainIndex[t][count[t]] = r[j]
    j = j + 1
  end
  if count:sum() == #classes * n then flg = false end
  
  i = i +1
end

torch.save('trainRDLIndex_' .. n ..'.t7',trainIndex:resize(n * #classes))
