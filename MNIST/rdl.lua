---------------
-- A Torch module for calculating the representational distance matrix (RDM) of a nn.module
-----
-- Author: Patrick McClure
-----
--Date Modified: 03/03/2015
---------------
--
--Input:
-- model -> An nn.module
-- inputs -> The data used to create the RDM
-- layers -> The layers of model for which RDMs are created
--
--Output:
-- rdm -> A #layers by #inputs*(#inputs-1)/2 Tensor with one RDM for each layer


require 'torch'
require 'nn'

rdl = {}

-- create a sum of squares RDM
function rdl.createSSRDM(model,inputs,layers)
-- initialize layers
layers = layers or torch.range(model:size())

-- initilize rdm
n = inputs:size(1)
local rdm = torch.Tensor(#layers,(n*n-n)/2)
rdm:zero()

local index = 0

-- iterate through every image pair
  for i = 1,inputs:size(1) do
    for j = i+1, inputs:size(1) do

      index = index+1
      
      -- iterate through every layer
      for l = 1,#layers  do
        k = layers[l]
        -- calculate representation for image i
        model:forward(inputs[i])
        local rep_i_k = model.modules[k].output:clone() -- :view(model.modules[k].output:nElement())
        
        -- calculate representation for image j
        model:forward(inputs[j])
        local rep_j_k = model.modules[k].output:clone() -- :view(model.modules[k].output:nElement())
        
        -- update rdm using the sum of squares distance
        rdm[l][index] = torch.pow(torch.dist(rep_i_k,rep_j_k),2)
      end
    end
    end
    return rdm
end

function rdl.createCosRDM(model,inputs,layers)
-- initialize layers
layers = layers or torch.range(model:size())

-- initialize rdm
n = inputs:size(1)
local rdm = torch.Tensor(#layers,(n*n-n)/2)
rdm:zero()

-- declare cosine distance module
cosDist = nn.CosineDistance()

local index = 0

-- iterate through every image pair
  for i = 1,inputs:size(1) do
    for j = i+1, inputs:size(1) do

      index = index+1

      -- iterate through every layer
      for l = 1,#layers  do
        k = layers[l]
        -- calculate representation for image i
        model:forward(inputs[i])
        local rep_i_k = model.modules[k].output:clone() -- :view(model.modules[k].output:nElement())
        
        -- calculate representation for image j
        model:forward(inputs[j])
        local rep_j_k = model.modules[k].output:clone() -- :view(model.modules[k].output:nElement())
        
        -- update rdm using the cosine distance
        rdm[l][index] = 1 + cosDist:forward({torch.reshape(rep_i_k,rep_i_k:nElement()),torch.reshape(rep_j_k,rep_j_k:nElement())})
    end
    end
    return rdm
end
end

function rdl.getExemplarPairs(indicies)
  
  local exemplarPairs = torch.Tensor(indicies:size(1)*(indicies:size(1)-1)/2,2)
  
  local index = 0
  
  for i = 1,indicies:size(1) do
    for j = i+1, indicies:size(1) do
      index = index + 1
      exemplarPairs[index][1] = indicies[i]
      exemplarPairs[index][2] = indicies[j]
    end
  end
  
  return exemplarPairs
end