# coding=utf-8

"""Processors for different tasks."""

import csv
import os

import pandas as pd

from util.tokenization import *


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
        
class Sentihood_NLI_M_Processor(DataProcessor):
    """Processor for the Sentihood data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_NLI_M.tsv"),sep="\t").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_NLI_M.tsv"),sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "test_NLI_M.tsv"),sep="\t").values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return ['None', 'Positive', 'Negative']

    def _create_examples(self, lines, set_type, debug=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
          #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[1]))
            text_b = convert_to_unicode(str(line[2]))
            label = convert_to_unicode(str(line[3]))
            if i==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
class PersentV1_Processor(DataProcessor):
    """Processor for the Sentihood data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_longformer_7topics.csv"))[['MASKED_DOCUMENT', 'sentiment', 'context']]
        return self._create_examples(train_data.values, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_longformer_7topics.csv"))[['MASKED_DOCUMENT', 'sentiment', 'context']]
        return self._create_examples(dev_data.values, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "random_test_longformer_7topics.csv"))[['MASKED_DOCUMENT', 'sentiment', 'context']]
        return self._create_examples(test_data.values, "test")
    
    def get_combo_examples(self, data_dir):
        test_data = pd.read_csv(os.path.join(data_dir, "random_test_longformer_7topics.csv"))[['MASKED_DOCUMENT', 'sentiment', 'context']]
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_longformer_7topics.csv"))[['MASKED_DOCUMENT', 'sentiment', 'context']]
        combo_data = pd.concat([test_data, dev_data])
        return self._create_examples(combo_data.values, "combo")
    
    def get_labels(self):
        """See base class."""
        return ['None', 'Neutral', 'Positive', 'Negative']

    def _create_examples(self, lines, set_type, debug=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
          #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = convert_to_unicode(str(line[2]))
            label = convert_to_unicode(str(line[1]))
            if i==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
class PersentV1_Para_Processor(DataProcessor):
    """Processor for the Sentihood data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_para_7aspects_128tokens.csv"))[['Paragraph', 'sentiment', 'context']]
        return self._create_examples(train_data.values, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_para_7aspects_128tokens.csv"))[['Paragraph', 'sentiment', 'context']]
        return self._create_examples(dev_data.values, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "random_test_para_7aspects_128tokens.csv"))[['Paragraph', 'sentiment', 'context']]
        return self._create_examples(test_data.values, "test")
    
    def get_combo_examples(self, data_dir):
        test_data = pd.read_csv(os.path.join(data_dir, "random_test_para_7aspects_128tokens.csv"))[['Paragraph', 'sentiment', 'context']]
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_para_7aspects_128tokens.csv"))[['Paragraph', 'sentiment', 'context']]
        combo_data = pd.concat([test_data, dev_data])
        return self._create_examples(combo_data.values, "combo")
    
    def get_labels(self):
        """See base class."""
        return ['None', 'Neutral', 'Positive', 'Negative']

    def _create_examples(self, lines, set_type, debug=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
          #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = convert_to_unicode(str(line[2]))
            label = convert_to_unicode(str(line[1]))
            if i==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
class PersentV1_Longformer_Processor(DataProcessor):
    """Processor for the Sentihood data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_longformer_7topics_auxiliary.csv"))[['MASKED_DOCUMENT', 'auxi_sent', 'group', 'label']]
        return self._create_examples(train_data.values, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_longformer_7topics_auxiliary.csv"))[['MASKED_DOCUMENT', 'auxi_sent', 'group', 'label']]
        return self._create_examples(dev_data.values, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "random_test_longformer_7topics_auxiliary.csv"))[['MASKED_DOCUMENT', 'auxi_sent', 'group', 'label']]
        return self._create_examples(test_data.values, "test")
    
    def get_combo_examples(self, data_dir):
        test_data = pd.read_csv(os.path.join(data_dir, "random_test_longformer_7topics_auxiliary.csv"))[['MASKED_DOCUMENT', 'auxi_sent', 'group', 'label']]
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_longformer_7topics_auxiliary.csv"))[['MASKED_DOCUMENT', 'auxi_sent', 'group', 'label']]
        combo_data = pd.concat([test_data, dev_data])
        return self._create_examples(combo_data.values, "combo")
    
    def get_labels(self):
        """See base class."""
        return ['0.0', '1.0']

    def _create_examples(self, lines, set_type, debug=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
          #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = convert_to_unicode(str(line[1]))
            text_c = convert_to_unicode(str(line[2]))
            label = convert_to_unicode(str(line[3]))
            if i==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("text_c=",text_c)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label))
        return examples

class Semeval_NLI_M_Processor(DataProcessor):
    """Processor for the Semeval 2014 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_NLI_M.csv"),header=None,sep="\t").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_NLI_M.csv"),header=None,sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "test_NLI_M.csv"),header=None,sep="\t").values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return ['positive', 'neutral', 'negative', 'conflict', 'none']

    def _create_examples(self, lines, set_type, debug=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
          #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[3]))
            text_b = convert_to_unicode(str(line[2]))
            label = convert_to_unicode(str(line[1]))
            if i==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class PersentV2_Processor(DataProcessor):
    """Processor for the Sentihood data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_7topics_noNeuMix.csv"))[['MASKED_DOCUMENT', 'sentiment', 'context']]
        return self._create_examples(train_data.values, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_7topics_noNeuMix.csv"))[['MASKED_DOCUMENT', 'sentiment', 'context']]
        return self._create_examples(dev_data.values, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "random_test_7topics_noNeuMix.csv"))[['MASKED_DOCUMENT', 'sentiment', 'context']]
        return self._create_examples(test_data.values, "test")
    
    def get_combo_examples(self, data_dir):
        test_data = pd.read_csv(os.path.join(data_dir, "random_test_7topics_noNeuMix.csv"))[['MASKED_DOCUMENT', 'sentiment', 'context']]
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_7topics_noNeuMix.csv"))[['MASKED_DOCUMENT', 'sentiment', 'context']]
        combo_data = pd.concat([test_data, dev_data])
        return self._create_examples(combo_data.values, "combo")
    
    def get_labels(self):
        """See base class."""
        return ['None', 'Positive', 'Negative']

    def _create_examples(self, lines, set_type, debug=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
          #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = convert_to_unicode(str(line[2]))
            label = convert_to_unicode(str(line[1]))
            if i==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class PersentV2_Para_Processor(DataProcessor):
    """Processor for the Sentihood data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_para_7topics_noNeuMix.csv"))[['Paragraph', 'Sentiment', 'Context']]
        return self._create_examples(train_data.values, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_para_7topics_noNeuMix.csv"))[['Paragraph', 'Sentiment', 'Context']]
        return self._create_examples(dev_data.values, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "random_test_para_7topics_noNeuMix.csv"))[['Paragraph', 'Sentiment', 'Context']]
        return self._create_examples(test_data.values, "test")
    
    def get_combo_examples(self, data_dir):
        test_data = pd.read_csv(os.path.join(data_dir, "random_test_para_7topics_noNeuMix.csv"))[['Paragraph', 'Sentiment', 'Context']]
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_para_7topics_noNeuMix.csv"))[['Paragraph', 'Sentiment', 'Context']]
        combo_data = pd.concat([test_data, dev_data])
        return self._create_examples(combo_data.values, "combo")
    
    def get_labels(self):
        """See base class."""
        return ['None', 'Positive', 'Negative']

    def _create_examples(self, lines, set_type, debug=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
          #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = convert_to_unicode(str(line[2]))
            label = convert_to_unicode(str(line[1]))
            if i==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
class PersentV2_Longformer_Processor(DataProcessor):
    """Processor for the Sentihood data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_longformer_7topics_auxiliary_noNeuMix.csv"))[['MASKED_DOCUMENT', 'auxi_sent', 'group', 'label']]
        return self._create_examples(train_data.values, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_longformer_7topics_auxiliary_noNeuMix.csv"))[['MASKED_DOCUMENT', 'auxi_sent', 'group', 'label']]
        return self._create_examples(dev_data.values, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "random_test_longformer_7topics_auxiliary_noNeuMix.csv"))[['MASKED_DOCUMENT', 'auxi_sent', 'group', 'label']]
        return self._create_examples(test_data.values, "test")
    
    def get_combo_examples(self, data_dir):
        test_data = pd.read_csv(os.path.join(data_dir, "random_test_longformer_7topics_auxiliary_noNeuMix.csv"))[['MASKED_DOCUMENT', 'auxi_sent', 'group', 'label']]
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_longformer_7topics_auxiliary_noNeuMix.csv"))[['MASKED_DOCUMENT', 'auxi_sent', 'group', 'label']]
        combo_data = pd.concat([test_data, dev_data])
        return self._create_examples(combo_data.values, "combo")
    
    def get_labels(self):
        """See base class."""
        return ['0.0', '1.0']

    def _create_examples(self, lines, set_type, debug=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
          #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[0]))
            text_b = convert_to_unicode(str(line[1]))
            text_c = convert_to_unicode(str(line[2]))
            label = convert_to_unicode(str(line[3]))
            if i==0 and debug:
                print(i)
                print("guid=",guid)
                print("text_a=",text_a)
                print("text_b=",text_b)
                print("text_c=",text_c)
                print("label=",label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=label))
        return examples

        