-- 
-- calculateSSRDM: Calculates the sum of squares RDM of an Torch NN model for a set of inputs
--
-- Author: Patrick McClure
-- Date  : 
-- Mod   : Feb 21, 2015
--
function rdl.calculateSSRDM(model, inputs, modelName, savePath)
-- initialize modelName and savePath if not passed
modelName = modelName or "model"
savePath = savePath or ""

-- declare and initilaize overall and temporary rdms
local num_layers = model:size();

local rdm = torch.Tensor(num_layers+1,inputs:size(1)*(inputs:size(1)-1)/2)
rdm:zero()
  
  --initialize index
  local index = 0
  -- iterate through every image pair
  for i = 1,inputs:size(1) do
    for j = i+1, inputs:size(1) do

      index = index+1

      for k = 1, model:size()  do
        
        -- calculate representation for image i
        model:forward(inputs[i])
        rep_i_k = model.modules[k].output:clone() -- :view(model.modules[k].output:nElement())
        
        -- calculate representation for image j
        model:forward(inputs[j])
        rep_j_k = model.modules[k].output:clone() -- :view(model.modules[k].output:nElement())
        
        -- update rdm using the sum of squares distance
        rdm[k][index] = rdm[k][index] + torch.pow(torch.dist(rep_i_k,rep_j_k),2)
        rdm[model:size()+1][index] = rdm[model:size()+1][index] + torch.pow(torch.dist(rep_i_k,rep_j_k),2)
      end
    end
  end
  
  -- save temporary rdm for the current model
  file = torch.DiskFile(savePath ..'rdm_' .. modelName .. '.asc', 'w')
  file:writeObject(rdm)
  file:close()
end

