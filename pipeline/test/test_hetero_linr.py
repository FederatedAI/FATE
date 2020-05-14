from pipeline.component.hetero_linr import HeteroLinR


a = HeteroLinR(name="hetero_linr_0", with_label=True)

print (a.output.data)
print (a.output.model)
