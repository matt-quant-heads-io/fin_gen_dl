"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import os
import argparse
import neat
import visualize
from neat.six_util import iteritems, itervalues
import json
import pika

parser = argparse.ArgumentParser(description='Kicks off the evolutionary search.')

# write queue
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
write_channel = connection.channel()
write_channel.exchange_declare(exchange='evo_search_exchange', exchange_type='direct')


if __name__ == '__main__':
    for hl_count in range(1, 6):
        msg = json.dumps({'hl_count': hl_count})
        write_channel.basic_publish(exchange='evo_search_exchange', routing_key="hl_{}".format(hl_count), body=msg)
