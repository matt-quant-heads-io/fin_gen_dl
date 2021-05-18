from __future__ import print_function
import neat

from neat.six_util import iteritems, itervalues
from model import compile_1_layer_model, compile_2_layer_model, compile_3_layer_model, compile_4_layer_model, compile_5_layer_model, compile_6_layer_model
from utils import create_train_test_set

import pandas as pd
import numpy as np

import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


def read_data(filename):
    df = pd.read_csv(f'{filename}.csv', header=0)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.sort_index(ascending=False, inplace=True)
    y = df.close
    df.drop(columns=['close'], inplace=True)
    column_names = list(df.columns)
    inputs = []
    outputs = []

    for idx in range(len(df)):
        inputs.append(tuple(df.iloc[idx, :].values))
        outputs.append(tuple([y.iloc[idx]]))

    return inputs, outputs


def get_col_names(filename):
    df = pd.read_csv(f'{filename}.csv', header=0)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.sort_index(ascending=False, inplace=True)
    y = df.close
    df.drop(columns=['close'], inplace=True)
    column_names = list(df.columns)
    return column_names


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        for xi, xo in zip(inputs, outputs):
            output = net.activate(xi)
            genome.fitness += -((output[0] - xo[0]) ** 2)


def run(config_file, inputs, outputs, local_dir):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    node_names = {}
    for i, col in enumerate(get_col_names(local_dir)):
        node_names[-i-1] = f"{col}"

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Plot starting topology from population:
    # =======================================
    init_network = None
    for g in itervalues(p.population):
        if init_network is None:
            init_network = g
            break

    # _ = visualize.draw_net(config, init_network, filename=f"init_network{hidden_layers}", view=True, node_names=node_names)


    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))


    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Return the winning genome.
    return winner, stats.fitness


test_dfs = ["test_df1_exp1", "test_df2_exp1", "test_df3_exp1"]
config_path = "configs/config-feedforward_{}.py"

results_dict = {1: [], 2: [], 3: [], 4: [], 5: []}
fitness_th = 0.90

# iterate thru the test dfs
for trial in range(30):
    for testdf in test_dfs:
        inputs, outputs = read_data(testdf)
        new_result = []
        for idx in range(1, 6):
            config_path = config_path.format(idx)
            winner, fitness = run(config_path, inputs, outputs, local_dir=testdf)
            new_result.append(fitness)
        new_res_idx = 0
        for res in new_result:
            new_res_idx += 1
            if res > fitness_th:
                results_dict[idx].append(winner)
                break

# print the results
for model_type, wins in results_dict.items():
    print(f"model_type: {model_type} , wins: {len(wins)} , avg loss: {sum(wins) / len(wins)}")

# Results:
# ========
# 1 layer: -32.786, 2 layer: -29.768, 3 layer: -12.708, 4 layer: -8.875, 5 layer: -6.714


# Tensorflow implementation
t_results_dict = {1: [], 2: [], 3: [], 4: [], 5: []}
epochscount = 10
model1_results = model2_results = model3_results = model4_results = model5_results = []
train_split_size = 0.8
for i in range(30):
    for testdf in test_dfs:
        for model_idx in range(1,6):
            K.clear_session()
            n_inputs = 1
            Xtrain, Xtest, ytrain, ytest, train_test_split = create_train_test_set(df=testdf, train_split=train_split_size)
            # setup model & TimeseriesGenerator, and train the model
            if model_idx == 1:
                model = compile_1_layer_model(n_inputs=n_inputs, n_features=len(Xtrain.shape[1]))
                generator = TimeseriesGenerator(Xtrain, ytrain, length=1, batch_size=1)
                model.fit_generator(generator, epochs=10)
                model1_results.append(model.history.history['loss'])
            if model_idx == 2:
                model = compile_2_layer_model(n_inputs=n_inputs, n_features=len(Xtrain.shape[1]))
                generator = TimeseriesGenerator(Xtrain, ytrain, length=1, batch_size=1)
                model.fit_generator(generator, epochs=10)
                model2_results.append(model.history.history['loss'])
            if model_idx == 3:
                model = compile_3_layer_model(n_inputs=n_inputs, n_features=len(Xtrain.shape[1]))
                generator = TimeseriesGenerator(Xtrain, ytrain, length=1, batch_size=1)
                model.fit_generator(generator, epochs=10)
                model3_results.append(model.history.history['loss'])
            if model_idx == 4:
                model = compile_4_layer_model(n_inputs=n_inputs, n_features=len(Xtrain.shape[1]))
                generator = TimeseriesGenerator(Xtrain, ytrain, length=1, batch_size=1)
                model.fit_generator(generator, epochs=10)
                model4_results.append(model.history.history['loss'])
            if model_idx == 5:
                model = compile_5_layer_model(n_inputs=n_inputs, n_features=len(Xtrain.shape[1]))
                generator = TimeseriesGenerator(Xtrain, ytrain, length=1, batch_size=1)
                model.fit_generator(generator, epochs=10)
                model5_results.append(model.history.history['loss'])


# Tally wins per model complexity
for loss_idx in range(len(model1_results)):
    temp_loss_arr = [model1_results[loss_idx], model2_results[loss_idx], model3_results[loss_idx], model4_results[loss_idx],
                     model5_results[loss_idx]]
    t_results_dict[np.argmin(temp_loss_arr)].append(1)

# Results:
# ========
# 1 layer: 0, 2 layer: 2, 3 layer: 7, 4 layer: 12, 5 layer: 69













