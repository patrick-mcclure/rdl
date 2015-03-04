---------------
-- A Torch module for calculating the representational distance matrix (RDM) of a nn.module
-----
-- Author: Patrick McClure
-----
--Date Modified: 03/03/2015
---------------

require 'torch'
require 'nn'

rdl = {}

function rdl.createSSRDM(model,inputs,layers)
layers = layers or torch.range(model:size())

n = inputs:size(1)

local rdm = torch.Tensor(#layers,(n*n-n)/2)
rdm:zero()

local index = 0
-- iterate through every image pair
  for i = 1,inputs:size(1) do
    for j = i+1, inputs:size(1) do

      index = index+1

      for l = 1,#layers  do
        k = layers[l]
        -- calculate representation for image i
        model:forward(inputs[i])
        local rep_i_k = model.modules[k].output:clone() -- :view(model.modules[k].output:nElement())
        
        -- calculate representation for image j
        model:forward(inputs[j])
        local rep_j_k = model.modules[k].output:clone() -- :view(model.modules[k].output:nElement())
        
        -- update rdm using the sum of squares distance
        rdm[l][index] = rdm[l][index] + torch.pow(torch.dist(rep_i_k,rep_j_k),2)
      end
    end
    end
    return rdm
end