from pipeline.component.hetero_pearson import HeteroPearson


a = HeteroPearson(name="hetero_pearson_0", with_label=True)

print (a.output.data)
print (a.output.model)
