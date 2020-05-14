from pipeline.component.sampler import FederatedSample


a = FederatedSample(name="federated_sample_0", with_label=True)

print (a.output.data)
print (a.output.model)
