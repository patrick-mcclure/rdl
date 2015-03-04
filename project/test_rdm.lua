require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'dataset-mnist'
require 'pl'
require 'paths'


file = torch.DiskFile('rdm.asc', 'r')
rdm = file:readObject()
file:close()