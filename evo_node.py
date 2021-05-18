from __future__ import print_function
import os, sys
import argparse
import neat
import visualize
from neat.six_util import iteritems, itervalues
import json
import pika
import pandas as pd


parser = argparse.ArgumentParser(description='Kicks off the evolutionary search.')
parser.add_argument('--hlayers', type=int, default=1)

# write queue
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
read_channel = connection.channel()
read_channel.exchange_declare(exchange='evo_search_exchange', exchange_type='direct')
write_channel = connection.channel()
write_channel.exchange_declare(exchange='best_model_exchange', exchange_type='direct')

# mechanism for reading in pandas df
df = pd.read_csv('mattsdatatest.csv', header=0)
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


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        for xi, xo in zip(inputs, outputs):
            output = net.activate(xi)
            print(f"output is {output}")
            genome.fitness += -(output[0] - xo[0]) ** 2


def run(config_file, hidden_layers, local_dir):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    node_names = {}
    for i, col in enumerate(column_names):
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

    _ = visualize.draw_net(config, init_network, filename=f"init_network{hidden_layers}", view=True, node_names=node_names)


    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Return the winning genome.
    return winner


if __name__ == '__main__':
    import sys
    local_dir = os.path.dirname(__file__)
    args = parser.parse_args()
    hlayers = args.hlayers
    hlayers = "hl_{}".format(hlayers)
    write_channel.queue_declare(queue="best_model", exclusive=False)
    write_channel.queue_bind(exchange="best_model_exchange", queue="best_model")


    ev_type = 1

    best_model = run(f'configs/config-feedforward_{ev_type}.py', ev_type, local_dir)
    print(f"best mode is {best_model}")


    # ORIGINAL CODE
    local_dir = os.path.dirname(__file__)
    args = parser.parse_args()
    hlayers = args.hlayers
    hlayers = "hl_{}".format(hlayers)
    # write_channel.queue_declare(queue="best_model", exclusive=False)
    # write_channel.queue_bind(exchange="best_model_exchange", queue="best_model")
    best_model = run('configs/config-feedforward_2.py', 2, local_dir)

    # def callback(ch, method, properties, body):
    #     input_msg = json.loads(body)
    #     config_path = os.path.join(local_dir, f'configs/config-feedforward_{input_msg["hl_count"]}.py')
    #     best_model = run(config_path, input_msg["hl_count"], local_dir)
    #     print(f'Loading config-feedforward_{input_msg["hl_count"]} for toplogical search with {input_msg["hl_count"]} hidden layers.')
    #     msg = json.dumps({'done': input_msg["hl_count"], 'winner': best_model['Fitness']})
    #     write_channel.basic_publish(exchange='best_model_exchange', routing_key="best_model", body=msg)
    #
    #
    #
    #
    # read_channel.queue_declare(queue=hlayers, exclusive=False)
    # read_channel.queue_bind(exchange='evo_search_exchange', queue=hlayers, routing_key=hlayers)
    #
    #
    # read_channel.basic_consume(queue=hlayers, on_message_callback=callback, auto_ack=True)
    #
    # print(' [*] Waiting for evolution trigger. To exit press CTRL+C')
    #
    # read_channel.start_consuming()

