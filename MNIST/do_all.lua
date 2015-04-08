require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'
require 'rdl'
require 'math'

opt = lapp[[
   -s,--save          (default "logs")      subdirectory to save logs
   -r,--learningRate  (default 0.05)        learning rate, for SGD only
   -b,--batchSize     (default 100)          batch size
   -m,--momentum      (default 0.75)           momentum
   -t,--threads       (default 4)           number of threads
   -n, --model_num   (default 1)           model number
]]

-- fix seed
torch.seed()

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

dofile 'data.lua'
dofile 'model.lua'
dofile 'train.lua'
dofile 'test.lua'
dofile 'rdm_train_and_test.lua'