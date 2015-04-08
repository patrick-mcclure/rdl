require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'
require 'rdl'
require 'math'

layers = {1,2,3}
nRDLTrain = 10
maxEpoch = 1

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train'.. opt.model_num ..'.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test'.. opt.model_num ..'.log'))

-- initialize epoch counter
epoch = 1

-- calculate and save initial model rdm
rdm = rdl.createSSRDM(model,trainRDL,layers)
torch.save('rdms/rdm_' .. opt.model_num .. '_' .. nRDLTrain .. '_' .. epoch-1 .. '.t7',rdm)
fbmat.save('rdms/rdm_' .. opt.model_num .. '_' .. nRDLTrain .. '_' .. epoch-1 .. '.mat',rdm)

while epoch < maxEpoch+1 do
   -- train/test
   train(trainData)
   test(testData)
   
   -- calculate and save rdm
   rdm = rdl.createSSRDM(model,trainRDL,layers)
   torch.save('rdms/rdm_' .. opt.model_num .. '_' .. nRDLTrain .. '_' .. epoch-1 .. '.t7',rdm)
   fbmat.save('rdms/rdm_' .. opt.model_num .. '_' .. nRDLTrain .. '_' .. epoch-1 .. '.mat',rdm)
   
   -- plot errors
  trainLogger:style{['% mean class accuracy (train set)'] = '-'}
  testLogger:style{['% mean class accuracy (test set)'] = '-'}
  trainLogger:plot()
  testLogger:plot()
end

-- save model and parameters
torch.save('models/model_' .. opt.model_num .. '.t7', model)
local weights_end, gradient_end = model:getParameters()
torch.save('vars/weights_' .. opt.model_num .. '.t7',weights_end)