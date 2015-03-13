require 'torch'
require 'nn'
require 'optim'
require 'dataset-mnist'
require 'pl'
require 'paths'
require 'rdl'

models = {1} --{1, 2, 3}

-- declare tensor of class labels
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {32,32}

-- declare set of layers of interest
layers = {3,6,10}

-- declare number of class exemplars
nRDLTrain = 10

-- setting training and test set sizes
nbTrainingPatches = 60000
nbTestingPatches = 10000

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- get the training images and targets
local inputs = torch.Tensor(trainData:size(1),1,geometry[1],geometry[2])
local targets = torch.Tensor(trainData:size(1))
local k = 1

-- declare and initilaize overall and temporary rdms
local rdm = torch.Tensor(#layers,nRDLTrain*#classes*(nRDLTrain*#classes-1)/2)
rdm:zero()

trainRDLIndex = torch.load('trainRDLIndex_' .. nRDLTrain .. '.t7')

trainRDL = torch.Tensor(trainRDLIndex:size(1),1,geometry[1],geometry[2])

for i = 1,trainRDLIndex:size(1) do
  local sample = trainData[trainRDLIndex[i]]
  trainRDL[i] = sample[1]:clone()
end

-- calculate the overall rdm and the rdms for each model
for m = 1,#models do
  model = torch.load('model_' .. models[m] .. '.t7'):double()
  
  model:evaluate()
  
  print('<model '.. models[m] .. '> loaded...')
  
  local rdm_tmp = rdl.createSSRDM(model,trainRDL,layers)
  
  -- update overall rdm
  rdm:add(rdm_tmp)
  
  -- save temporary rdm for the current model
  torch.save('rdm' .. models[m].. '_' .. nRDLTrain .. '.t7', rdm_tmp)
  
  -- reinitialize temporary rdm
  rdm_tmp:zero()
  
  print('<model '.. models[m] .. '> rdm calculated')
end
-- save the overall rdm
torch.save('rdm_ensemble' .. nRDLTrain .. '.t7',rdm)