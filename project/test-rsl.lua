require 'torch'
require 'nn'
require 'rdl'
require 'dataset-mnist'

geometry = {32,32}

model = torch.load('model_1.t7'):double()

trainRDLIndex = torch.load('trainRDLIndex_5.t7')

layers = {3,6,10}


trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

local inputs = torch.Tensor(trainRDLIndex:size(1),1,geometry[1],geometry[2])
      local k = 1
      for i = 1,trainRDLIndex:size(1) do
         -- load new sample
         local sample = trainData[trainRDLIndex[i]]
         local input = sample[1]:clone()
         inputs[k] = input
         k = k + 1
      end
  
  rdm = rdl.createSSRDM(model,inputs,layers)
  
  print('DONE!')






