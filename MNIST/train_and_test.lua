require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'
require 'rdl'
require 'math'

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train'.. opt.model_num ..'.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test'.. opt.model_num ..'.log'))

-- initialize epoch counter
epoch = 1
maxEpoch = 1

while epoch < maxEpoch+1 do
   -- train/test
   train(trainData)
   test(testData)
   
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