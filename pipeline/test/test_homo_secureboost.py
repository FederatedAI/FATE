from pipeline.component.homo_secureboost import HomoSecureBoost


a = HomoSecureBoost(name="homo_secureboost_0", with_label=True)

print (a.output.data)
print (a.output.model)
