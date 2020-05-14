from pipeline.component.evaluation import Evaluation


a = Evaluation(name="evaluation_0", with_label=True)

print (a.output.data)
print (a.output.model)
