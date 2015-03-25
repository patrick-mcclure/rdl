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
      stage[3]:add(nn.View(-1,#classes))
      stage[3]:add(nn.LogSoftMax())
      
      tmp1 = nn.Sequential()
      tmp1:add(nn.View(-1,64*2*2))
      tmp1:add(nn.MulConstant(1))
      
      tmp2 = nn.Sequential()
      tmp2:add(nn.View(-1,32*9*9))
      tmp2:add(nn.MulConstant(1))
      
      -- add auxilary classifiers
      aux = {}
      
      aux[1] = nn.Concat(2)
      aux[1]:add(stage[3])
      aux[1]:add(tmp1)
      
      stage[2]:add(aux[1])

      aux[2] = nn.Concat(2)
      aux[2]:add(stage[2])
      aux[2]:add(tmp2)

      -- build model
      model = nn.Sequential()
      model:add(stage[1])
      model:add(aux[2])
      
      -- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

input = torch.Tensor(2, 1, 32,32)

input[1] = trainData[1][1]
input[2] = trainData[2][1]

output = model:forward(input)
z = torch.Tensor(output:size()):zero()
model:backward(input,z:contiguous())
print(1)