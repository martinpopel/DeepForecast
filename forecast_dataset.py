import csv
import logging
import numpy as np

class ForecastDataset:
    """Class capable of loading forecast datasets in csv format."""

    def __init__(self, filename, train=None):
        """Load dataset from file in csv format.

        Arguments:
        train: If given, the products and projects name->id mappings
               from the training data will be reused.
        """

        # Initialize name<->id mappings.
        self.proj_map = train.proj_map if train else {'unk': 0}
        self.prod_map = train.prod_map if train else {'unk': 0}
        self.projects = train.projects if train else ['unk']
        self.products = train.products if train else ['unk']

        # Load the data from csv.
        self.data = []
        self.skipped = 0
        with open(filename, newline='') as file:
            csvreader = csv.reader(file)
            # skip header line
            csvreader.__next__()
            for line_n, row in enumerate(csvreader, start=1):
                problems = {'Inf', 'NaN', 'NA'}.intersection(row)
                if problems:
                    if self.skipped == 0:
                        logging.warning("Skipping an example with %s %s:%d\n%s",
                                        problems, filename, line_n, str(row))
                    self.skipped += 1
                    continue
                example = {}
                # columns 1-12 ... previous months consumptions
                # column 13 ... id of the month to be predicted
                example['features'] = row[1:14]
                prod_name, proj_name = row[25:27]
                if prod_name not in self.prod_map:
                    self.prod_map[prod_name] = len(self.products)
                    self.products.append(prod_name)
                if proj_name not in self.proj_map:
                    self.proj_map[proj_name] = len(self.projects)
                    self.projects.append(proj_name)
                example['prod_id'] = self.prod_map[prod_name]
                example['proj_id'] = self.proj_map[proj_name]
                example['gold'] = row[-1]
                self.data.append(example)
        self.data_size = len(self.data)
        logging.warning("%d examples read from %s. %d examples skipped.",
                        self.data_size, filename, self.skipped)
        self._permutation = np.random.permutation(self.data_size)

    def features_size(self):
        return 13

    def next_batch(self, batch_size):
        """Return the next batch.

        Arguments:
        batch_size: how many examples should be in the batch

        Returns: (features, prod_id, proj_id, gold)
        """

        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]
        return self._next_batch(batch_perm)

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(self.data_size)
            return True
        return False

    def whole_data_as_batch(self):
        """Return the whole dataset in the same result as next_batch.

        Returns the same type of results as next_batch.
        """
        return self._next_batch(np.arange(self.data_size))

    def _next_batch(self, batch_perm):
        batch_size = len(batch_perm)
        features = np.zeros([batch_size, self.features_size()])
        gold = np.zeros([batch_size])
        prod_id = np.zeros([batch_size])
        proj_id = np.zeros([batch_size])
        for i in range(batch_size):
            example = self.data[batch_perm[i]]
            features[i, :] = example['features']
            prod_id[i] = example['prod_id']
            proj_id[i] = example['proj_id']
            gold[i] = example['gold']
        return features, prod_id, proj_id, gold
