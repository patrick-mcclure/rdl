require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'
require 'rdl'
require 'math'

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

torch.setdefaulttensortype('torch.FloatTensor')

-- initialize fbmattorch instance
fbmat = require 'fb.mattorch'

-- define output classes
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {32,32}

-- set dataset variables
nbTrainingPatches = 60000
nbTestingPatches = 10000

-- load training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- load test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- load RDL data
nRDLTrain = 10

trainRDLIndex = torch.load('vars/trainRDLIndex_' .. nRDLTrain .. '.t7'):double()

if isRDL then
  -- create auxilary training set
auxBatchSize = 50
-- load auxiliary dataset 
auxInputs = rdl.getExemplarPairs(trainRDLIndex)
auxTargets = torch.load('rdms/rdm_ensemble' .. nRDLTrain .. '.t7'):double()
auxTargets:div(auxTargets:norm())
end

trainRDL = torch.Tensor(trainRDLIndex:size(1),1,geometry[1],geometry[2])

for i = 1,trainRDLIndex:size(1) do
  local sample = trainData[trainRDLIndex[i]]
  trainRDL[i] = sample[1]:clone()
end