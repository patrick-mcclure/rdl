require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'
require 'rdl'
require 'math'

-- training function
function train(dataset)
  -- epoch tracker
  epoch = epoch or 1

   -- local vars
  local time = sys.clock()
  
   -- set model to training mode
  model:training()

  local randIndex = torch.randperm(dataset:size())
  
  local rep = nn.Replicate(opt.batchSize)
  
   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[randIndex[i]]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()
        
        -- calculate auxilary error
        local auxError1 = torch.Tensor(auxBatchSize,32, 9, 9):zero()
        local auxError2 = torch.Tensor(auxBatchSize,64, 2, 2):zero()
        local auxTargets1 = torch.Tensor(auxBatchSize)
        local auxTargets2 = torch.Tensor(auxBatchSize)
        local dist1 = torch.Tensor(auxBatchSize):zero()
        local dist2 = torch.Tensor(auxBatchSize):zero()
        local alpha = alpha or opt.alpha
        local index = 0
        
        local auxIndecies = torch.randperm(auxInputs:size(1))
        
        for i = t,math.min(t+auxBatchSize-1,auxIndecies:size(1)) do
        index = index + 1
          local auxIndex = auxIndecies[i]
         	
          local imgIndex_1 = auxInputs[auxIndex][1]
          local imgIndex_2 = auxInputs[auxIndex][2]
          
          auxTargets1[index] = auxTargets[1][auxIndex]
          auxTargets2[index] = auxTargets[2][auxIndex]
          
          local input1 = dataset[imgIndex_1][1]:clone()
         	local input2 = dataset[imgIndex_2][1]:clone()
          
         	model:forward(input1)
          output11 = model.modules[1].output:clone()
          output12 = model.modules[2].output:clone()
          
          model:forward(input2)
          output21 = model.modules[1].output:clone()
          output22 = model.modules[2].output:clone()
          
          dist1[index] = auxCriterion:forward(output11,output21)
          auxError1[index]:add(auxCriterion:backward(output11,output21))
          
          dist2[index] = auxCriterion:forward(output12,output22)
          auxError2[index]:add(auxCriterion:backward(output12,output22))
        end
        
        -- calculate target betas
        local beta1 = dist1:norm()
        
        local beta2 = dist2:norm()
        
        -- scale target distance
        local dif1 = torch.mul(auxTargets1,beta1)
        local dif2 = torch.mul(auxTargets2, beta2)
        
        dif1:add(-1,dist1)
        dif2:add(-1,dist2)
        
        dif1:mul((1-alpha))
        dif2:mul((1-alpha))
        
        -- calculate errors
        local auxE1 = torch.Tensor(32,9,9)
        auxE1:zero()
        local auxE2 = torch.Tensor(64,2,2)
        auxE2:zero()
        for i = 1,auxBatchSize do
           if dif1[i] == -0 then
             dif1[i] = 0
           end
           if dif2[i] == -0 then
             dif2[i] = 0
           end
         auxE1:add(dif1[i],auxError1[i]:clone())
         auxE2:add(dif2[i],auxError2[i]:clone())
        end
        
        -- calculate gradient per main mini-batch input
        auxE1:div(inputs:size(1))
        auxE2:div(inputs:size(1))
        
        if auxE2:sum() ~= 0 then
          print('HELP!')
        end
         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         --model:backward(inputs, df_do)
         local outputs1 = stage[1].output:clone()
         local outputs2 = stage[2].output:clone()
         local error2 = stage[3]:backward(outputs2, df_do)
         local error1 = stage[2]:backward(outputs1,torch.add(error2,rep:forward(auxE2)))
         stage[1]:backward(inputs,torch.add(error1,rep:forward(auxE1)))

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(outputs[i], targets[i])
         end

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
         -- Perform SGD step:
         sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = 5e-7
         }
         optim.sgd(feval, parameters, sgdState)
      
         -- disp progress
         xlua.progress(t, dataset:size())
   end
   
   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'mnist.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('<trainer> saving network to '..filename)
   -- torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
   end