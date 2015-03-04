require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'

net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 96, 11, 11, 4, 4))

print(net.modules[1].output)