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

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -s,--save          (default "rdl_logs")  subdirectory to save logs
   -n,--network       (default "")          reload pretrained network
   -m,--model         (default "rdl")       type of model tor train: rdl
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
]]

-- fix seed
torch.seed()

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

   torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {32,32}

--      stage = {}
   
--      -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
--      stage[1] = nn.Sequential()
--      stage[1]:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
--      stage[1]:add(nn.ReLU())
--      stage[1]:add(nn.SpatialMaxPooling(3, 3, 3, 3))
--      -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
--      stage[2] = nn.Sequential()
--      stage[2]:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
--      stage[2]:add(nn.ReLU())
--      stage[2]:add(nn.SpatialMaxPooling(2, 2, 2, 2))

--      -- stage 3 : standard 2-layer MLP:
--      stage[3] = nn.Sequential()
--      stage[3]:add(nn.Reshape(64*2*2))
--      stage[3]:add(nn.Linear(64*2*2, 500))
--      stage[3]:add(nn.ReLU())
--      stage[3]:add(nn.Linear(500, #classes))
--      stage[3]:add(nn.View(-1,#classes))
--      stage[3]:add(nn.LogSoftMax())
      
      
--      tmp1 = nn.Sequential()
--      tmp1:add(nn.View(-1,64*2*2))
--      tmp1:add(nn.MulConstant(1))
      
--      tmp2 = nn.Sequential()
--      tmp2:add(nn.View(-1,32*9*9))
--      tmp2:add(nn.MulConstant(1))
      
--      -- add auxilary classifiers
--      aux = {}
      
--      aux[1] = nn.Concat(2)
--      aux[1]:add(stage[3])
--      aux[1]:add(tmp1)
      
--      stage[2]:add(aux[1])

--      aux[2] = nn.Concat(2)
--      aux[2]:add(stage[2])
--      aux[2]:add(tmp2)

--      -- build model
      model = nn.Sequential()
--      model:add(stage[1])
--      model:add(aux[2])

-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
      -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 3 : standard 2-layer MLP:
      model:add(nn.Reshape(64*2*2))
      --model:add(nn.Dropout())
      model:add(nn.Linear(64*2*2, 500))
      model:add(nn.ReLU())
      model:add(nn.Linear(500, #classes))
      model:add(nn.LogSoftMax())

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<mnist> using model:')
print(model)

----------------------------------------------------------------------
-- loss functions: class -> negative log-likelihood and aux -> squared distance
--

criterion = nn.ClassNLLCriterion()
auxCriterion = nn.MSECriterion()

----------------------------------------------------------------------
-- get/create dataset
--
   nbTrainingPatches = 60000
   nbTestingPatches = 10000

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
local auxIndecies = torch.randperm(auxInputs:size(1))
local auxTargets = torch.load('rdms/rdm_ensemble' .. nExemplars .. '.t7')

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
   
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
        
         -- initialize classification output and error variables
         local n_aux_1 = 32*9*9
         local n_aux_2 = 64*2*2
         
         local df_do = torch.Tensor(opt.batchSize,#classes+n_aux_1+n_aux_2)
         df_do:zero()
         local df_do_class = df_do:narrow(2,1,#classes)
         local df_do_aux_1 = df_do:narrow(2,#classes+1+n_aux_2,n_aux_1)
         local df_do_aux_2 = df_do:narrow(2,#classes+1,n_aux_2) 
         local f_class = 0  
         local alpha = 1

         local df_do_1 = torch.Tensor(n_aux_1)
         df_do_1:zero()
         local df_do_2 = torch.Tensor(n_aux_2)
         df_do_2:zero()
           
         local index = 1

--        -- find auxilary RDMs
--         for i = t,math.min(t+auxBatchSize-1,auxIndecies:size(1)) do
         	
--          local auxIndex = auxIndecies[i]
         	
--          local imgIndex_1 = auxInputs[auxIndex][1]
--          local imgIndex_2 = auxInputs[auxIndex][2]
          
--          local input1 = dataset[imgIndex_1][1]:clone()
--         	local input2 = dataset[imgIndex_2][1]:clone()

--         	output1 = model:forward(input1):clone()
--         	output2 = model:forward(input2):clone()
          
--          local aux_outputs11 = output1:narrow(2,#classes+1+n_aux_2,n_aux_1):clone()
--          local aux_outputs12 = output1:narrow(2,#classes+1,n_aux_2):clone()
--          local aux_outputs21 = output2:narrow(2,#classes+1+n_aux_2,n_aux_1):clone()
--          local aux_outputs22 = output2:narrow(2,#classes+1,n_aux_2):clone()
          
--         	local d1 = auxCriterion:forward(aux_outputs11,aux_outputs21)
--          df_do_1:add((1-alpha)*(auxTargets[1][auxIndex]-d1), auxCriterion:backward(aux_outputs11,aux_outputs21))
                    
--         	local d2 = auxCriterion:forward(aux_outputs12,aux_outputs22)
--          df_do_2:add((1-alpha)*(auxTargets[2][auxIndex]-d2), auxCriterion:backward(aux_outputs12,aux_outputs22))
         
--          index = index + 1
--         end
            
--         df_do_1:div(inputs:size(1))
--         df_do_2:div(inputs:size(1))
        
         local outputClass = torch.Tensor(opt.batchSize,#classes)
--         -- evaluate function for complete mini batch
        
--           df_do_aux_1[i]:copy(df_do_1)
--           df_do_aux_2[i]:copy(df_do_2)
           
          local outputs = model:forward(inputs)
          --outputClass:copy(output:narrow(2,1,#classes))
           -- update total error calculation
           f_class = criterion:forward(outputs, targets)
        
           -- estimate df/dW
           df_do_class:copy(criterion:backward(outputs, targets))
           
      
        --df_do_class:mul(alpha)
       
        model:backward(inputs,df_do_class) --df_do)
        
        for i = 1,inputs:size(1) do
        confusion:add(outputs[i], targets[i])
        end
         
         local f = f_class

        --trainLogger:add{['Average auxilary distance error (train set)'] = f_aux/auxBatchSize}

         -- return f and df/dX
         return f,gradParameters
      end


         -- Perform SGD step:
         sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = 5e-7
         }
         local q,pq = feval
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

      outputs = model:forward(inputs)
      outputsClass = outputs:narrow(1,1,#classes)
      -- confusion:
      for i = 1,outputs:size(1) do
         confusion:add(outputsClass[i], targets[i])
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