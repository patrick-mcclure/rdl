require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'
require 'rdl'
require 'math'

-- define convolutional network model
-----------------------------------------------------------------------

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


-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- print model structure
print('<mnist> using model:')
print(model)

-- set loss function to negative log-likelihood
criterion = nn.ClassNLLCriterion()