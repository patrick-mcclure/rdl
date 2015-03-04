require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'

models = {1} --{1, 2, 3}

-- declare tensor of class labels
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {32,32}

-- setting training and test set sizes
nbTrainingPatches = 2000
nbTestingPatches = 1000

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- get the training images and targets
local inputs = torch.Tensor(trainData:size(1),1,geometry[1],geometry[2])
local targets = torch.Tensor(trainData:size(1))
local k = 1

local rdm_input_to_mnist_map = torch.Tensor(trainData:size(1))

for i = 1, trainData:size(1) do
  -- create rdm input to mnist map
  rdm_input_to_mnist_map[k] = i

  -- load new sample
  local sample = trainData[i]
  local input = sample[1]:clone()
  local _,target = sample[2]:clone():max(1)
  target = target:squeeze()
  inputs[k] = input
  targets[k] = target
  k = k + 1
end

-- save the targets for the training samples
local = torch.DiskFile('targets.asc', 'w')
model = file:writeObject(targets)
file:close()

-- declare and initilaize overall and temporary rdms
local layers = torch.range(model:size());
local n = inputs:size(1)*(inputs:size(1)-1)/2

local rdm = torch.Tensor(num_layers+1,n)
rdm:zero()

local rdm_tmp = torch.Tensor(num_layers+1,n)

-- create training
local rdm_input_indicies = torch.Tensor(n)

-- calculate the overall rdm and the rdms for each model
for m = 1,#models do
  local file = torch.DiskFile('model_' .. models[m] .. '.asc', 'r')
  model = file:readObject():double()
  file:close()
  
  model:evaluate()
  
  print('<model '.. models[m] .. '> loaded...')
  
  --initialize index
  local index = 0
  
  -- iterate through every image pair
  for i = 1,inputs:size(1) do
    for j = i+1, inputs:size(1) do

      index = index+1
      
      if m == models[1] then
        rdm_input_indicies[index][1] = rdm_input_to_mnist_map[i]
        rdm_input_indicies[index][2] = rdm_input_to_mnist_map[j]
      end

      for k = 1, model:size()  do
        
        -- calculate representation for image i
        model:forward(inputs[i])
        rep_i_k = model.modules[k].output:clone() -- :view(model.modules[k].output:nElement())
        
        -- calculate representation for image j
        model:forward(inputs[j])
        rep_j_k = model.modules[k].output:clone() -- :view(model.modules[k].output:nElement())
        
        -- update rdm using the sum of squares distance
        rdm_tmp[k][index] = rdm_tmp[k][index] + torch.pow(torch.dist(rep_i_k,rep_j_k),2)
        rdm_tmp[model:size()+1][index] = rdm_tmp[model:size()+1][index] + torch.pow(torch.dist(rep_i_k,rep_j_k),2)
      end
    end
  end
  
  -- update overall rdm
  rdm:add(rdm_tmp)
  
  -- save temporary rdm for the current model
  torch.save('rdm' .. models[m] .. '.t7', 'w')
  
  -- reinitialize temporary rdm
  rdm_tmp:zero()
  
  print('<model '.. models[m] .. '> rdm calculated')
end
-- save the overall rdm
torch.save('rdm.t7',rdm)

-- save the rdm input indicies
torch.save('rdm_input_indicies.t7', rdm_input_indicies)