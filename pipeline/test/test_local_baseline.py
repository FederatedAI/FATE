from pipeline.component.local_baseline import LocalBaseline


a = LocalBaseline(name="local_baseline_0", with_label=True)

print (a.output.data)
print (a.output.model)
