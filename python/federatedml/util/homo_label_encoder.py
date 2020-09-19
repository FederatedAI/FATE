from federatedml.transfer_variable.transfer_class.homo_label_encoder_transfer_variable \
    import HomoLabelEncoderTransferVariable
from federatedml.util import consts
from federatedml.util import LOGGER


class HomoLabelEncoderClient(object):

    def __init__(self):
        self.transvar = HomoLabelEncoderTransferVariable()

    def label_alignment(self, class_set):

        LOGGER.info('start homo label alignments')
        self.transvar.local_labels.remote(class_set, role=consts.ARBITER, suffix=('label_align',))
        new_label_mapping = self.transvar.label_mapping.get(idx=0, suffix=('label_mapping',))
        reverse_mapping = {v: k for k, v in new_label_mapping.items()}
        new_classes_index = [new_label_mapping[k] for k in new_label_mapping]
        new_classes_index = sorted(new_classes_index)
        aligned_labels = [reverse_mapping[i] for i in new_classes_index]
        return aligned_labels, new_label_mapping


class HomoLabelEncoderArbiter(object):

    def __init__(self):
        self.transvar = HomoLabelEncoderTransferVariable()

    def label_alignment(self):
        LOGGER.info('start homo label alignments')
        labels = self.transvar.local_labels.get(idx=-1, suffix=('label_align', ))
        label_set = set()
        for local_label in labels:
            label_set.update(local_label)
        global_label = list(label_set)
        global_label = sorted(global_label)
        label_mapping = {v: k for k, v in enumerate(global_label)}
        self.transvar.label_mapping.remote(label_mapping, idx=-1, suffix=('label_mapping',))
        return label_mapping
