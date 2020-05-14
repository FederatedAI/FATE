from pipeline.component.dataio import DataIO


a = DataIO(name="dataio_0", with_label=True)

print (a.output.data)
print (a.output.model)
