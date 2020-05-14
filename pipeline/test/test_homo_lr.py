from pipeline.component.homo_lr import HomoLR


a = HomoLR(name="homo_lr_0", with_label=True)

print (a.output.data)
print (a.output.model)
