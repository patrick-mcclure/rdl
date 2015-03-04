require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'

models = {1, 2, 3}

-- declare tensor of class labels
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {32,32}

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

trainSetSize = 10; -- trainData:size(1)

local inputs = torch.Tensor(trainData:size(1)/2000,1,geometry[1],geometry[2])
local k = 1
for i = 1, trainData:size(1), 2000 do
  -- load new sample
  local sample = trainData[i]
  local input = sample[1]:clone()
  inputs[k] = input
  k = k + 1
end

local rdm = torch.Tensor(inputs:size(1)*(inputs:size(1)-1)/2)

rdm:zero()

local rdm_tmp = torch.Tensor(inputs:size(1)*(inputs:size(1)-1)/2)

rdm_tmp:zero()

for m = 1,#models do
  file = torch.DiskFile('model_' .. models[m] .. '.asc', 'r')
  model = file:readObject():double()
  file:close()

  local l = 0

  for i = 1,inputs:size(1) do
    for j = i+1, inputs:size(1) do

      l = l+1
      print("l = " .. l)
      for k = 1, model:size()  do
        model:forward(inputs[i])
        rep_i_k = model.modules[k].output:clone() -- :view(model.modules[k].output:nElement())
        
        model:forward(inputs[j])
        rep_j_k = model.modules[k].output:clone() -- :view(model.modules[k].output:nElement())
        rdm_tmp[l] = rdm_tmp[l] + torch.pow(torch.dist(rep_i_k,rep_j_k),2)
        print(torch.pow(torch.dist(rep_i_k,rep_j_k),2))
      end
    end
  end
  
  rdm = rdm + rdm_tmp
  
  file = torch.DiskFile('rdm' .. models[m] .. '.asc', 'w')
  file:writeObject(rdm_tmp)
  file:close()
  
  rdm_tmp:zero()
  
end
file = torch.DiskFile('rdm.asc', 'w')
file:writeObject(rdm / #models)
file:close()