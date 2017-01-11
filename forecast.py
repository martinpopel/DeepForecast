#!/usr/bin/env python3

import datetime
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses

import forecast_dataset

class Network:
    def __init__(self, args, data_train):
        self.args = args

        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = args.seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(
            inter_op_parallelism_threads=args.threads,
            intra_op_parallelism_threads=args.threads))

        # Construct the graph
        with self.session.graph.as_default():
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            # [None] is the shape, gold is a 1-dimensional tensor, the dimension is the batch size
            self.gold = tf.placeholder(tf.float32, [None], "gold_consumption")
            self.prod_id = tf.placeholder(tf.int32, [None], "product_id")
            self.proj_id = tf.placeholder(tf.int32, [None], "project_id")
            # consumption from previous months etc
            self.features = tf.placeholder(tf.float32, [None, data_train.features_size()],
                                           "features")

            n_prods = len(data_train.products)
            n_projs = len(data_train.projects)
            prod_embeddings_matrix = tf.get_variable("prod_embeddings_matrix",
                                                     [n_prods, args.prod_dim], dtype=tf.float32)
            proj_embeddings_matrix = tf.get_variable("proj_embeddings_matrix",
                                                     [n_projs, args.proj_dim], dtype=tf.float32)
             # [batch_size, prod_dim] = float32
            prod_embeddings = tf.nn.embedding_lookup(prod_embeddings_matrix, self.prod_id)
            proj_embeddings = tf.nn.embedding_lookup(proj_embeddings_matrix, self.proj_id)

            # concat all the inputs along the 1st dimension (keep the 0th dimension as batch_size)
            layer = tf.concat(1, [prod_embeddings, proj_embeddings, self.features])

            # TODO try more layers, other activation_fns, dropout,...
            if args.hidden_dim:
                layer = tf_layers.fully_connected(layer, num_outputs=args.hidden_dim,
                                                  activation_fn=tf.nn.relu, scope="hidden_layer")

            layer = tf_layers.linear(layer, num_outputs=1, scope="output_layer")
            self.predictions = tf.reshape(layer, [-1])

            differences = self.predictions - self.gold
            self.mse = tf.reduce_mean(tf.square(differences))
            # TODO real quantile_loss with argparter tau, now it is just the median
            self.quantile = tf_losses.absolute_difference(self.predictions, self.gold)

            # TODO try optimizing a different loss
            optimizer = tf.train.AdamOptimizer()
            loss = self.mse if args.loss == 'mse' else self.quantile
            #self.gradient = optimizer.compute_gradients(loss)
            # TODO: report the gradient to summaries, do gradient clipping
            #self.training = optimizer.apply_gradients(self.gradient, global_step=self.global_step)
            self.training = optimizer.minimize(loss, global_step=self.global_step)

            self.train_summary = tf.summary.merge(
                [tf.summary.scalar("train/mse", self.mse),
                 tf.summary.scalar("train/quantile", self.quantile)])
            self.dev_summary = tf.summary.merge(
                [tf.summary.scalar("dev/mse", self.mse),
                 tf.summary.scalar("dev/quantile", self.quantile)])

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            if args.expname is None:
                args.expname = "proj{}-prod{}-hid{}-{}-bs{}-epochs{}".format(
                    args.proj_dim, args.prod_dim, args.hidden_dim, args.loss, args.batch_size,
                    args.epochs)
            self.summary_writer = tf.summary.FileWriter("{}/{}-{}".format(
                args.logdir, timestamp, args.expname), graph=self.session.graph, flush_secs=10)

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, data_train):
        features, prod_id, proj_id, gold = data_train.next_batch(self.args.batch_size)
        feed_dict = {self.prod_id: prod_id, self.proj_id: proj_id, self.gold: gold,
                     self.features: features}
        _, summ = self.session.run([self.training, self.train_summary], feed_dict)
        self.summary_writer.add_summary(summ, self.training_step)

    def evaluate(self, data_dev):
        features, prod_id, proj_id, gold = data_dev.whole_data_as_batch()
        feed_dict = {self.prod_id: prod_id, self.proj_id: proj_id, self.gold: gold,
                     self.features: features}
        mse, quantile, summ = self.session.run([self.mse, self.quantile, self.dev_summary],
                                               feed_dict)
        self.summary_writer.add_summary(summ, self.training_step)
        return mse, quantile


    def predict(self, data_test):
        features, prod_id, proj_id, _ = data_test.whole_data_as_batch()
        feed_dict = {self.prod_id: prod_id, self.proj_id: proj_id,
                     self.features: features}
        return self.session.run(self.predictions, feed_dict)

def main():
    import argparse
    argpar = argparse.ArgumentParser()
    # General arguments
    argpar.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    argpar.add_argument("--data_train", default="data/tableTrainingImputed.csv",
                        type=str, help="Training data file.")
    argpar.add_argument("--data_dev", default="data/tableTestingImputed.csv",
                        type=str, help="Development data file.")
    argpar.add_argument("--data_test", default="data/tableTestingImputed.csv",
                        type=str, help="Test data file.")
    argpar.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    argpar.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    argpar.add_argument("--threads", default=0, type=int, help="Number of threads to use. 0=all.")
    argpar.add_argument("--seed", default=42, type=int, help="Random seed.")
    argpar.add_argument("--expname", default=None, type=str, help="Experiment name.")

    # Project-specific arguments
    argpar.add_argument("--loss", default='mse', type=str, help="Optimization loss (mse,quantile).")
    argpar.add_argument("--proj_dim", default=20, type=int, help="Project embeddings dimension.")
    argpar.add_argument("--prod_dim", default=20, type=int, help="Product embeddings dimension.")
    argpar.add_argument("--hidden_dim", default=20, type=int, help="Hidden layer dimension.")
    args = argpar.parse_args()

    # Fix random seed
    np.random.seed(args.seed)

    print("Loading the data.", file=sys.stderr)
    data_train = forecast_dataset.ForecastDataset(args.data_train)
    data_dev = forecast_dataset.ForecastDataset(args.data_dev, data_train)
    data_test = forecast_dataset.ForecastDataset(args.data_test, data_train)

    print("Constructing the network.", file=sys.stderr)
    network = Network(args, data_train=data_train)

    # Train
    best_dev_mse = float('Inf')
    test_predictions = None

    print("Training...", file=sys.stderr)
    for epoch in range(args.epochs):
        while not data_train.epoch_finished():
            network.train(data_train)

        dev_mse, dev_quantile = network.evaluate(data_dev)
        print("Epoch {:3d}: dev mse={:.4f} quantile_loss={:.4f}".format(
            epoch + 1, dev_mse, dev_quantile), file=sys.stderr)

        if dev_mse < best_dev_mse:
            best_dev_mse = dev_mse
            test_predictions = network.predict(data_test)

    # Print test predictions
    for i in range(len(data_test.data)):
        print("{:.6f}".format(test_predictions[i]))

if __name__ == "__main__":
    main()
