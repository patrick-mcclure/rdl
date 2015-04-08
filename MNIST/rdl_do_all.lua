require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'
require 'rdl'
require 'math'
require 'rdl_train_norm'
require 'rdl_train_regression'

opt = lapp[[
   -s,--save          (default "rdl_logs")      subdirectory to save logs
   -r,--learningRate  (default 0.05)        learning rate, for SGD only
   -b,--batchSize     (default 100)          batch size
   -m,--momentum      (default 0.75)           momentum
   -t,--threads       (default 4)           number of threads
   -n, --model_num   (default "regression")           model number
   -a, --alpha        (default 0.99)           relative learning rate for RDL
]]

-- fix seed
torch.seed()

torch.setdefaulttensortype('torch.FloatTensor')

-- set isRDL flag
isRDL = true

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

dofile 'data.lua'

dofile 'model.lua'

auxCriterion = nn.MSECriterion()

if opt.model_num == "regression" then
  dofile 'rdl_train_regression.lua'
else
  dofile 'rdl_train_norm.lua'
end

dofile 'test.lua'

dofile 'train_and_test.lua'