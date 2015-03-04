function rdl.writeFile(object,name)

file = torch.DiskFile(name .. '.asc', 'r')
object = file:writeObject(object)
file:close()