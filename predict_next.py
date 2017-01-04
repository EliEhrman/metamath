# import numpy as np
# import tensorflow as tf
import os
import csv

data_dir = '/devlink/data/metamath/'
runup_num = 5

fh_labels = open(data_dir+'setshrunk.txt')
icsv_labels = csv.reader(fh_labels, delimiter=',')
label_params = {row[0]: row[1:] for row in icsv_labels}
fh_labels.close()

fh_proofs = open(data_dir+'prooflists.txt')
icsv_proofs = csv.reader(fh_proofs, delimiter=',')
proofs = [row for row in icsv_proofs]
fh_proofs.close()

targets = [row[0] for row in proofs]
proof_bodies = [row[1:] for row in proofs]

proof_body_len = len(proof_bodies[0])

for itarget, target in enumerate(targets):
	for ibod, bod in enumerate(proof_bodies[itarget]):
		inputvals = label_params[target]
		outputvals = label_params[bod]
		for runup in range(ibod-runup_num, ibod):
			if runup < 0:
				inputvals += [0.0 in range(proof_body_len)]
			else:
				inputvals += label_params[proof_bodies[runup]]


print('bye')



