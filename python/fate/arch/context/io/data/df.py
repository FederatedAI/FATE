class Dataframe:
    def __init__(self, frames, num_features, num_samples) -> None:
        self.data = frames
        self.num_features = num_features
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def to_local(self):
        return self.data.to_local()
