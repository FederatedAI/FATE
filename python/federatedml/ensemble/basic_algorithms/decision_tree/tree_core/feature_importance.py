class FeatureImportance(object):

    def __init__(self, importance=0, importance_2=0, main_type='split'):

        self.legal_type = ['split', 'gain']
        assert main_type in self.legal_type, 'illegal importance type {}'.format(main_type)
        self.importance = importance
        self.importance_2 = importance_2
        self.main_type = main_type

    def add_gain(self, val):
        if self.main_type == 'gain':
            self.importance += val
        else:
            self.importance_2 += val

    def add_split(self, val):
        if self.main_type == 'split':
            self.importance += val
        else:
            self.importance_2 += val

    def from_protobuf(self, feature_importance):
        self.main_type = feature_importance.main
        self.importance = feature_importance.importance
        self.importance_2 = feature_importance.importance2
        if self.main_type == 'split':
            self.importance = int(self.importance)

    def __cmp__(self, other):

        if self.importance > other.importance:
            return 1
        elif self.importance < other.importance:
            return -1
        else:
            return 0

    def __eq__(self, other):
        return self.importance == other.importance

    def __lt__(self, other):
        return self.importance < other.importance

    def __repr__(self):
        return 'importance type: {}, importance: {}, importance2 {}'.format(self.main_type, self.importance,
                                                                            self.importance_2)

    def __add__(self, other):
        new_importance = FeatureImportance(main_type=self.main_type, importance=self.importance + other.importance,
                                           importance_2=self.importance_2 + other.importance_2)
        return new_importance
