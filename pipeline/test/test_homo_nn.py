from pipeline.component.homo_nn import HomoNN


a = HomoNN(name="homo_nn_0", with_label=True)

print (a.output.data)
print (a.output.model)
