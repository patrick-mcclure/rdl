require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'

-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {32,32}

stage = {}
   
      -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
      stage[1] = nn.Sequential()
      stage[1]:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
      stage[1]:add(nn.Tanh())
      stage[1]:add(nn.SpatialMaxPooling(3, 3, 3, 3))

      -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
      stage[2] = nn.Sequential()
      stage[2]:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
      stage[2]:add(nn.Tanh())
      stage[2]:add(nn.SpatialMaxPooling(2, 2, 2, 2))

      -- stage 3 : standard 2-layer MLP:
      stage[3] = nn.Sequential()
      stage[3]:add(nn.Reshape(64*2*2))
      stage[3]:add(nn.Linear(64*2*2, 200))
      stage[3]:add(nn.Tanh())
      stage[3]:add(nn.Linear(200, #classes))
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
      
      -- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

input = trainData[1][1]

output = model:forward(input)
print(output:nElement())

local check_3 = model.modules[2].modules[2].output
local check_2 = model.modules[2].modules[1].modules[4].modules[2].output
local check_1 = model.modules[2].modules[1].modules[4].modules[1].output

print(1)
