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
   -b,--batchSize     (default 10)          batch size
   -m,--momentum      (default 0)           momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 4)           number of threads
]]

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats, for SGD
if opt.optimization == 'SGD' then
   torch.setdefaulttensortype('torch.FloatTensor')
end

-- batch size?
if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
   error('LBFGS should not be used with small mini-batches; 1000 is a recommended')
end

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

--   if opt.model == 'convnet' then
--      ------------------------------------------------------------
--      -- convolutional network 
--      ------------------------------------------------------------
--      -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
--      model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
--      model:add(nn.Tanh())
--      model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
--      -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
--      model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
--      model:add(nn.Tanh())
--      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
--      -- stage 3 : standard 2-layer MLP:
--      model:add(nn.Reshape(64*2*2))
--      model:add(nn.Linear(64*2*2, 200))
--      model:add(nn.Tanh())
--      model:add(nn.Linear(200, #classes))
--      ------------------------------------------------------------

   if opt.model == 'rdl' then
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
      
      -- add auxilary classifiers
      aux = {}
      
      aux[1] = nn.Concat(1)
      aux[1]:add(stage[3])
      aux[1]:add(nn.Reshape(64*2*2))
      
      stage[2]:add(aux[1])

      aux[2] = nn.Concat(1)
      aux[2]:add(stage[2])
      aux[2]:add(nn.Reshape(32*9*9))

      -- build model
      model = nn.Sequential()
      model:add(stage[1])
      model:add(aux[2])

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
-- loss functions: class -> negative log-likelihood and aux -> squared distance
--

criterion = nn.ClassNLLCriterion()
auxCriterion = nn.MSECriterion()

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
auxBatchSize = 10
-- load auxiliary dataset 
       local file = torch.DiskFile('rdm_input_indicies.asc', 'r')
       auxInputs = file:readObject():double()
       file:close()
       auxIndecies = torch.randperm(auxInputs:size(1))
       
       file = torch.DiskFile('rdm.asc', 'r')
       auxTargets = file:readObject():double()
       file:close()

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
         local sample = dataset[i]
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
         
         local df_do = torch.Tensor(#classes+n_aux_1+n_aux_2)
         df_do:zero()
         local df_do_class = df_do:narrow(1,1,#classes)
         local df_do_aux_1 = df_do:narrow(1,#classes+1+n_aux_2,n_aux_1)
         local df_do_aux_2 = df_do:narrow(1,#classes+1,n_aux_2) 
         local f_class = 0  
         local f_aux = 0
         
         local targetRDM = torch.Tensor(2,auxBatchSize)
         
         local dEa_dy_1 = torch.Tensor(auxBatchSize,n_aux_1)
         local dEa_dy_2 = torch.Tensor(auxBatchSize,n_aux_2)
         local rdm_aux_1 = torch.Tensor(auxBatchSize)
         local rdm_aux_2 = torch.Tensor(auxBatchSize)
         local index = 1
        
        -- find auxilary RDMs
         for i = t,math.min(t+auxBatchSize-1,auxIndecies:size(1)) do
         	
          local auxIndex = auxIndecies[i]
          targetRDM[1][i] = auxTargets[3][i]
          targetRDM[2][i] = auxTargets[6][i]
         	
          local imgIndex_1 = auxInputs[auxIndex][1]
          local imgIndex_2 = auxInputs[auxIndex][2]
          
          local input1 = dataset[imgIndex_1][1]:clone()
         	local input2 = dataset[ingIndex_2][1]:clone()

         	output1 = model:forward(input1)
         	output2 = model:forward(input2)
          
          local aux_outputs11 = output1:narrow(1,#classes+1+n_aux_2+1,n_aux_1)
          local aux_outputs12 = df_do:narrow(1,#classes+1,n_aux_2) 
          local aux_outputs21 = output2:narrow(1,#classes+1+n_aux_2+1,n_aux_1)
          local aux_outputs22 = df_do:narrow(1,#classes+1,n_aux_2) 
          
         	rdm_aux_1[index] = imageDistance - auxCriterion:forward(aux_outputs11,aux_outputs21)
         	rdm_aux_2[index] = imageDistance - auxCriterion:forward(aux_outputs12,aux_outputs22)
          
          dEa_dy_1[index] = auxCriterion:backward(aux_outputs11,aux_outputs21)
          dEa_dy_2[index] = auxCriterion:backward(aux_outputs12,aux_outputs22)
         
          index = index + 1
         end
        
        -- calculate auxilary RDM-based gradients
      local dist_error_1 = targetRDM[1]:add(-1,rdm_aux_1)
      local dist_error_2 = targetRDM[2]:add(-1,rdm_aux_2)
      
      f_aux = dist_error_1:pow(2):sum(1) + dist_error_2:pow(2):sum(1)
      
      df_do_aux_1 = torch.mm(dEa_dy_1:transpose(1,2),dist_error_1) 
      df_do_aux_2 = torch.mm(dEa_dy_2:transpose(1,2),dist_error_1)
      
      df_do_aux_1:div(inputs:size(1))
      df_do_aux_2:div(inputs:size(1))
        
         -- evaluate function for complete mini batch
         for i = 1,inputs:size(1) do
          local output = model:forward(inputs[i])
          local outputClass = output:narrow(1,1,#classes)
           -- update total error calculation
           f_class = f_class + criterion:forward(outputClass, targets[i])
        
           -- estimate df/dW
           df_do_class = criterion:backward(outputClass, targets[i])
          
          model:backward(inputs[i], df_do)
        end
        
         
         local f = f_class/inputs:size(1) + f_aux/auxBatchSize
         

         -- penalties (L1 and L2):
         if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign

            -- Loss:
            f = f + opt.coefL1 * norm(parameters,1)
            f = f + opt.coefL2 * norm(parameters,2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
         end

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(outputsClassifier[i], targets[i])
         end

        trainLogger:add{['Average auxilary distance error (train set)'] = f_aux/auxBatchSize}

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
         local q,pq = feval
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
         confusion:add(preds[i]:narrow(1,1,#classes+1), targets[i])
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
while true do
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
