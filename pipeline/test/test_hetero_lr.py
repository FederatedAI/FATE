from pipeline.component.hetero_lr import HeteroLR


a = HeteroLR(name="hetero_lr_0", with_label=True)

print (a.output.data)
print (a.output.model)
