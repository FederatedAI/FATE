from pipeline.component.hetero_nn import HeteroNN


a = HeteroNN(name="hetero_nn_0", with_label=True)

print (a.output.data)
print (a.output.model)
