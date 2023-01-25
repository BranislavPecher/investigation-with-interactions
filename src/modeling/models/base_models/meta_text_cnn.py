from src.modeling.models.base_models.abstract_meta_base_model import AbstractMetaBaseModel
import torch
import torch.nn as nn

from collections import OrderedDict
from torchmeta.modules import MetaConv2d, MetaBatchNorm2d, MetaSequential, MetaLinear


def conv_text_block(in_channels, out_channels, kernel_size, pool_size, **kwargs):
    return MetaSequential(OrderedDict([
        ('convolutional', MetaConv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)),
        # ('norm', nn.BatchNorm2d(out_channels, momentum=1., track_running_stats=False)),
        ('relu', nn.ReLU()),
        ('squeeze', nn.Flatten(2, -1)),
        ('pool', nn.MaxPool1d(pool_size))
    ]))


class DenseClassifier(AbstractMetaBaseModel):
    def __init__(self, sentence_length, embedding_dim, n_filters, filter_size, pool_size, hidden_size, num_classes, init_classifier=True):
        super(DenseClassifier, self).__init__()
        self.sentence_length = sentence_length
        self.embedding_dim = embedding_dim

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.init_classifier = init_classifier
        print(f'Initialise classifier set to {self.init_classifier}')

        self.size = sentence_length * embedding_dim
        self.dense = MetaSequential(OrderedDict([
                                                 ('dense', MetaLinear(self.size, hidden_size, bias=True)),
                                                 ('relu', nn.ReLU())
        ]))
        self.classifier = MetaLinear(self.hidden_size, num_classes, bias=True)

    def forward(self, embedded_text, params=None):
        embedded = embedded_text.view(-1, self.size)
        x = self.dense(embedded, params=self.get_subdict(params, 'dense'))
        logits = self.classifier(x, params=self.get_subdict(params, 'classifier'))
        return logits


class MetaSimpleCnnTextProto(AbstractMetaBaseModel):
    def __init__(self, sentence_length, embedding_dim, n_filters, filter_size, pool_size, hidden_size, num_classes, init_classifier=True):
        super(MetaSimpleCnnTextProto, self).__init__()
        self.in_channels = 1
        self.sentence_length = sentence_length
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.init_classifier = init_classifier
        print(f'Initialise classifier set to {self.init_classifier}')

        self.convolution = conv_text_block(self.in_channels, n_filters, (self.filter_size, embedding_dim), pool_size)

        size = int((sentence_length - self.filter_size + 1) / pool_size)
        self.dense = MetaSequential(OrderedDict([
                                                 ('dense', MetaLinear(n_filters * size, hidden_size, bias=True))
                                                #  ('relu', nn.ReLU())
        ]))

    def forward(self, embedded_text, params=None):
        embedded = embedded_text.unsqueeze(1)
        convolution = self.convolution(embedded, params=self.get_subdict(params, 'convolution0'))
        x = convolution.view(convolution.shape[0], -1)
        x = self.dense(x, params=self.get_subdict(params, 'dense'))
        return x

class MetaSimpleCnnText(AbstractMetaBaseModel):
    def __init__(self, sentence_length, embedding_dim, n_filters, filter_size, pool_size, hidden_size, num_classes, init_classifier=True):
        super(MetaSimpleCnnText, self).__init__()
        self.in_channels = 1
        self.sentence_length = sentence_length
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.init_classifier = init_classifier
        print(f'Initialise classifier set to {self.init_classifier}')

        self.convolution = conv_text_block(self.in_channels, n_filters, (self.filter_size, embedding_dim), pool_size)

        size = int((sentence_length - self.filter_size + 1) / pool_size)
        self.dense = MetaSequential(OrderedDict([
                                                 ('dense', MetaLinear(n_filters * size, hidden_size, bias=True)),
                                                 ('relu', nn.ReLU())
        ]))
        if self.init_classifier:
            self.classifier = MetaLinear(hidden_size, num_classes, bias=True)

    def forward(self, embedded_text, params=None):
        embedded = embedded_text.unsqueeze(1)
        convolution = self.convolution(embedded, params=self.get_subdict(params, 'convolution0'))
        x = convolution.view(convolution.shape[0], -1)
        x = self.dense(x, params=self.get_subdict(params, 'dense'))
        if self.init_classifier:
            logits = self.classifier(x, params=self.get_subdict(params, 'classifier'))
            return logits
        else:
            return x


class MetaTextCnn(AbstractMetaBaseModel):
    def __init__(self, sentence_length, embedding_dim, n_filters, filter_sizes, pool_size, hidden_size, num_classes):
        super(MetaTextCnn, self).__init__()
        self.in_channels = 1
        self.sentence_length = sentence_length
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.pool_size = pool_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.convolution0 = conv_text_block(self.in_channels, n_filters, (filter_sizes[0], embedding_dim), pool_size)
        self.convolution1 = conv_text_block(self.in_channels, n_filters, (filter_sizes[1], embedding_dim), pool_size)
        self.convolution2 = conv_text_block(self.in_channels, n_filters, (filter_sizes[2], embedding_dim), pool_size)
        size = 0
        for filter_size in filter_sizes:
            size += int((sentence_length - filter_size + 1) / pool_size)
        self.dense = MetaSequential(OrderedDict([
                                                 ('dense', MetaLinear(n_filters * size, hidden_size, bias=True)),
                                                 ('relu', nn.ReLU())
        ]))
        self.classifier = MetaLinear(hidden_size, num_classes, bias=True)

    def forward(self, embedded_text, params=None):
        embedded = embedded_text.unsqueeze(1)
        convolution0 = self.convolution0(embedded, params=self.get_subdict(params, 'convolution0'))
        convolution1 = self.convolution1(embedded, params=self.get_subdict(params, 'convolution1'))
        convolution2 = self.convolution2(embedded, params=self.get_subdict(params, 'convolution2'))

        concatenation = torch.cat([convolution0, convolution1, convolution2], dim=2)
        x = concatenation.view(concatenation.shape[0], -1)
        x = self.dense(x, params=self.get_subdict(params, 'dense'))
        logits = self.classifier(x, params=self.get_subdict(params, 'classifier'))
        return logits
