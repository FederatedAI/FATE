from pipeline.component.union import Union


a = Union(name="union_0", with_label=True)

print (a.output.data)
print (a.output.model)
