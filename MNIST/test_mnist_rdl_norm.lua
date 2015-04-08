----------------------------------------------------------------------
-- This script shows how to train different models on the MNIST 
-- dataset, using multiple optimization techniques (SGD, LBFGS)
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'
require 'rdl'
require 'math'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -s,--save          (default "rdm_logs")      subdirectory to save logs
   -n,--network       (default "")          reload pretrained network
   -m,--model         (default "convnet")   type of model tor train: convnet | mlp | linear
   -f,--full                                use the full dataset
   -p,--plot                                plot while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 0.05)        learning rate, for SGD only
   -b,--batchSize     (default 100)          batch size
   -m,--momentum      (default 0.75)           momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 4)           number of threads
   -k, --model_num    (default 3)           model number
]]

opt.full = true

-- fix seed
torch.seed()

nRDLTrain = 10

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats, for SGD
if opt.optimization == 'SGD' then
   torch.setdefaulttensortype('torch.FloatTensor')
end

-- initialize fbmattorch instance
local fbmat = require 'fb.mattorch'
----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {32,32}

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()

   if opt.model == 'convnet' then
            stage = {}
   
      -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
      stage[1] = nn.Sequential()
      stage[1]:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
      stage[1]:add(nn.ReLU())
      stage[1]:add(nn.SpatialMaxPooling(3, 3, 3, 3))
      -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
      stage[2] = nn.Sequential()
      stage[2]:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
      stage[2]:add(nn.ReLU())
      stage[2]:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      
      -- stage 3 : standard 2-layer MLP:
      stage[3] = nn.Sequential()
      stage[3]:add(nn.Reshape(64*2*2))
      stage[3]:add(nn.Linear(64*2*2, 500))
      stage[3]:add(nn.ReLU())
      stage[3]:add(nn.Linear(500, #classes))
      stage[3]:add(nn.LogSoftMax())
      
      model = nn.Sequential()
      model:add(stage[1])
      model:add(stage[2])
      model:add(stage[3])
      ------------------------------------------------------------

   else
      print('Unknown model type')
      cmd:text()
      error()
   end
else
   print('<trainer> reloading previously trained network')
   model = torch.load(opt.network)
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<mnist> using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
--model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
auxCriterion = nn.MSECriterion()
rep = nn.Replicate(opt.batchSize)

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
else
   nbTrainingPatches = 2000
   nbTestingPatches = 1000
   print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

-- create auxilary training set
auxBatchSize = 50
-- load auxiliary dataset 
local nExemplars = 10
local trainRDLIndex = torch.load('vars/trainRDLIndex_' .. nExemplars ..'.t7'):double()
local auxInputs = rdl.getExemplarPairs(trainRDLIndex)
local auxTargets = torch.load('rdms/rdm_ensemble' .. nExemplars .. '.t7')
auxTargets:div(auxTargets:norm())

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

trainRDLIndex = torch.load('vars/trainRDLIndex_' .. nRDLTrain .. '.t7')

trainRDL = torch.Tensor(trainRDLIndex:size(1),1,geometry[1],geometry[2])

for i = 1,trainRDLIndex:size(1) do
  local sample = trainData[trainRDLIndex[i]]
  trainRDL[i] = sample[1]:clone()
end

-- training function
function train(dataset)
  -- epoch tracker
  epoch = epoch or 1

   -- local vars
  local time = sys.clock()
   
   -- set model to training mode
  model:training()

  local randIndex = torch.randperm(dataset:size())
  
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
        local alpha = 0.97
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
        
--        if auxE2:sum() ~= 0 then
--          print('HELP!')
--        end
        

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

--        local error2 = stage[3]:backward(outputs2, df_do)
--        local error1 = stage[2]:backward(outputs1,error2)
--        stage[1]:backward(inputs,error1)

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(outputs[i], targets[i])
         end

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      if opt.optimization == 'LBFGS' then

         -- Perform LBFGS step:
         lbfgsState = lbfgsState or {
            maxIter = opt.maxIter,
            lineSearch = optim.lswolfe
         }
         optim.lbfgs(feval, parameters, lbfgsState)
       
         -- disp report:
         print('LBFGS step')
         print(' - progress in batch: ' .. t .. '/' .. dataset:size())
         print(' - nb of iterations: ' .. lbfgsState.nIter)
         print(' - nb of function evalutions: ' .. lbfgsState.funcEval)

      elseif opt.optimization == 'SGD' then

         -- Perform SGD step:
         sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = 5e-7
         }
         optim.sgd(feval, parameters, sgdState)
      
         -- disp progress
         xlua.progress(t, dataset:size())

      else
         error('unknown optimization method')
      end
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
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   -- torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

-- test function
function test(dataset)
   
   -- local vars
   local time = sys.clock()
  
    -- set model to evaluate mode
  model:evaluate()
  
   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- test samples
      local preds = model:forward(inputs)

      -- confusion:
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()
end

----------------------------------------------------------------------
-- and train!
--

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train'.. opt.model_num ..'.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test'.. opt.model_num ..'.log'))

epoch = 1
while epoch < 2 do
   -- train/test
   train(trainData)
   test(testData)
   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
end