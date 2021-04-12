
from __future__ import print_function
import os
import argparse
import neat
import visualize
from neat.six_util import iteritems, itervalues
import json
import pika

parser = argparse.ArgumentParser(description='Kicks off the evolutionary search.')
parser.add_argument('--hlayers', type=int, default=1)

# write queue
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
read_channel = connection.channel()
read_channel.exchange_declare(exchange='evo_search_exchange', exchange_type='direct')

# Create the mechanism for reading in pandas df
inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(inputs, outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2


def run(config_file, hidden_layers, local_dir):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    node_names = {-1: 'x_1', -2: 'x_2', 0: 'act_1'}

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

    # Display the winning genome.
    print(f'\nBest genome for hidden layers={hidden_layers}:\n{winner}')

    # Show output of the most fit genome against training data.
    print(f'\nOutput from hidden layers={hidden_layers}:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(inputs, outputs):
        output = winner_net.activate(xi)
        print(f"hidden layers={hidden_layers}: input {xi}, expected output {xo}, got {output}")

    _ = visualize.draw_net(config, winner, filename=f"best_hl_{hidden_layers}", view=True, node_names=node_names)

    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint(os.path.join(local_dir, 'checkpoints/neat-checkpoint-4'))
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    args = parser.parse_args()
    hlayers = args.hlayers
    hlayers = "hl_{}".format(hlayers)

    def callback(ch, method, properties, body):
        input_msg = json.loads(body)
        config_path = os.path.join(local_dir, f'configs/config-feedforward_{input_msg["hl_count"]}.py')
        run(config_path, input_msg["hl_count"], local_dir)
        print(f'Loading config-feedforward_{input_msg["hl_count"]} for toplogical search with {input_msg["hl_count"]} hidden layers.')


    read_channel.queue_declare(queue=hlayers, exclusive=False)
    read_channel.queue_bind(exchange='evo_search_exchange', queue=hlayers, routing_key=hlayers)

    read_channel.basic_consume(queue=hlayers, on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for evolution trigger. To exit press CTRL+C')

    read_channel.start_consuming()

