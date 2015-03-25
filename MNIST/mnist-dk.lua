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

-- set soft target temperature
local Temp = 20

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- get the training images and targets
local inputs = torch.Tensor(trainData:size(1),1,geometry[1],geometry[2])
local targets = torch.Tensor(trainData:size(1))
local k = 1
for i = 1, trainData:size(1) do
  -- load new sample
  local sample = trainData[i]
  local input = sample[1]:clone()
  local _,target = sample[2]:clone():max(1)
  target = target:squeeze()
  inputs[k] = input
  targets[k] = target
  k = k + 1
end

-- declare and initilaize the model outputs
local outputs = torch.Tensor(#models+4,inputs:size(1),#classes)
outputs:zero()
outputs[#models+3] = torch.ones(inputs:size(1),#classes)

-- calculate the soft targets for each input
for m = 1,#models do
  model = torch.load('models/model_' .. m .. '.t7'):double()

  print('<model '.. models[m] .. '> loaded...')
  
  model:evaluate()
  
  -- iterate through every image pair
  for i = 1,inputs:size(1) do
    outputs[m][i] = model:forward(inputs[i])
    outputs[#models+1][i]:add(torch.exp(outputs[m][i]))
    outputs[#models+3][i]:cmul(torch.exp(outputs[m][i]))
  end
  print('<model '.. models[m] .. '> outputs calculated')
end

outputs[#models+1]:div(#models)
outputs[#models+2] = torch.log(outputs[#models+1])

-- finish calculating the geometric mean of the models outputs
outputs[#models+3] = torch.pow(outputs[#models+3],1/#models)

-- calculate the softened targets for each input
soft = nn.SoftMax()
for i = 1,inputs:size(1) do
outputs[#models+4][i] = soft:forward(torch.div(outputs[#models+3][i],Temp))
end
-- save the model outputs and targets_dk
torch.save('vars/outputs_all.t7', outputs)
print('<outputs_all> saved')

torch.save('vars/outputs_avg.t7', outputs[#models+1])
print('<outputs_avg> saved')

torch.save('vars/outputs_ensemble.t7', outputs[#models+2])
print('<outputs_ensemble> saved')

torch.save('vars/outputs_geo.t7', outputs[#models+3])
print('<outputs_geo> saved')

torch.save('vars/soft_targets.t7',outputs[#models+4])
print('<soft_targets> saved')