from pipeline.component.one_hot_encoder import OneHotEncoder


a = OneHotEncoder(name="one_hot_encoder_0", with_label=True)

print (a.output.data)
print (a.output.model)
