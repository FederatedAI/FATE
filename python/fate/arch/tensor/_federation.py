class FederationDeserializer:
    def do_deserialize(self, ctx, party):
        ...

    @classmethod
    def make_frac_key(cls, base_key, frac_key):
        return f"{base_key}__frac__{frac_key}"
