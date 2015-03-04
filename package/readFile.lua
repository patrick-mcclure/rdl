function rdl.readFile(name)

file = torch.DiskFile(name .. '.asc', 'r')
object = file:readObject()
file:close()
return object