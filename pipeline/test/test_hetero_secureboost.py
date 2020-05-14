from pipeline.component.hetero_secureboost import HeteroSecureBoost


a = HeteroSecureBoost(name="hetero_secureboost_0", with_label=True)

print (a.output.data)
print (a.output.model)
