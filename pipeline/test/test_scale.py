from pipeline.component.scale import FeatureScale


a = FeatureScale(name="scale_0", with_label=True)

print (a.output.data)
print (a.output.model)
