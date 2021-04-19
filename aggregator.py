
from __future__ import print_function
import os
import argparse
import sys
import neat
import visualize
from neat.six_util import iteritems, itervalues
import json
import pika
import numpy as np

parser = argparse.ArgumentParser(description='Kicks off the evolutionary search.')
parser.add_argument('--acc_th', type=float, default=0.85)

# write queue
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
read_channel = connection.channel()
read_channel.exchange_declare(exchange='best_model_exchange', exchange_type='direct')

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    args = parser.parse_args()
    acc_th = args.acc_th
    # global msg_count, best_model_acc, best_model
    #
    # msg_count = 0
    # best_model_acc = -np.inf
    # best_model = None


    def callback(ch, method, properties, body):
        print(f"Inside of aggregator.py")
        # msg_count += 1
        # input_msg = json.loads(body)
        # if input_msg['winner'] > best_model_acc:
        #     best_model = {'hl_size': input_msg['done'], 'fitness': input_msg['winner']}
        #     best_model_acc = input_msg['winner']
        # if msg_count == 5:
        #     print(f"The best model was from evo hlayers={input_msg['done']} w/ fitness={input_msg['winner']}")
        #     sys.exit(0)

    read_channel.queue_declare(queue="best_model", exclusive=False)
    read_channel.queue_bind(exchange='best_model_exchange', queue="best_model", routing_key="best_model")
    read_channel.basic_consume(queue="best_model", on_message_callback=callback, auto_ack=True)

    print(' [*] Aggregator queue. To exit press CTRL+C')

    read_channel.start_consuming()
