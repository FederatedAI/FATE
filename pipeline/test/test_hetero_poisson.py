from pipeline.component.hetero_poisson import HeteroPoisson


a = HeteroPoisson(name="hetero_poisson_0", with_label=True)

print (a.output.data)
print (a.output.model)
