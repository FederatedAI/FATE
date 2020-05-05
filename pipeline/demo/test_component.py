from pipeline.component.component_base import Component

a = Component(name="test")
b = a.get_party_instance(role='guest', party_id=1)
bb = a.get_party_instance(role='guest', party_id=[1,2,3,4])
c = Component()

print (a.name)
print (b.name)
print (bb.name)
print (c.name)
