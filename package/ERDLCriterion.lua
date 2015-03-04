require "nn"

local ERDLCriterion, parent = torch.class('rdl.ERDLCriterion', 'nn.Criterion')

function ERDLCriterion:__init()
   parent.__init(self)
   self.signedError = 0
end

function ERDLCriterion:updateOutput(input1, input2, RDMDistance)
   return input.rdl.ERDLCriterion_updateOutput(self, input1, input2, RDMDistance)
end

function ERDLCriterion:updateGradInput(input1, input2, RDMDistance)
   return input.rdl.ERDLCriterion_updateGradInput(self, input1, input2, RDMDistance)
end