import numpy as np
import tensorflow as tf
import os
import csv
import random
import math
import matplotlib.pyplot as plt

data_dir = '/devlink/data/metamath/'
tfchk_fname = 'predict.chk'
runup_num = 5
batch_size = 256
num_learn_iters = 1000000


def inference(input, batch_size, input_rec_len, num_hidden1_units, num_hidden2_units, output_rec_len):
	# Hidden 1
	with tf.name_scope('hidden1'):
		weights = tf.Variable(
			tf.truncated_normal([input_rec_len, num_hidden1_units],
								stddev=1.0 / math.sqrt(float(input_rec_len))),
			name='weights')
		biases = tf.Variable(tf.zeros([num_hidden1_units]),
							 name='biases')
		hidden1 = tf.nn.relu(tf.matmul(input, weights) + biases)
	# Hidden 2
	with tf.name_scope('hidden2'):
		weights = tf.Variable(
			tf.truncated_normal([num_hidden1_units, num_hidden2_units],
								stddev=1.0 / math.sqrt(float(num_hidden1_units))),
			name='weights')
		biases = tf.Variable(tf.zeros([num_hidden2_units]),
							 name='biases')
		hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
	# Linear
	with tf.name_scope('softmax_linear'):
		weights = tf.Variable(
			tf.truncated_normal([num_hidden2_units, output_rec_len],
								stddev=1.0 / math.sqrt(float(num_hidden2_units))),
			name='weights')
		biases = tf.Variable(tf.zeros([output_rec_len]),
							 name='biases')
		logits = tf.matmul(hidden2, weights) + biases
	return logits


def loss(logits, output):
	"""Calculates the loss from the logits and the labels.

	Args:
	  logits: Logits tensor, float - [batch_size, NUM_CLASSES].
	  labels: Labels tensor, int32 - [batch_size].

	Returns:
	  loss: Loss tensor of type float.

	"""
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(logits, output)), axis=1))
	return loss

def init_data():
	fh_labels = open(data_dir+'setshrunk.txt')
	icsv_labels = csv.reader(fh_labels, delimiter=',')
	label_params = {row[0]: [float(sval) for sval in row[1:]] for row in icsv_labels}
	fh_labels.close()

	fh_proofs = open(data_dir+'prooflists.txt')
	icsv_proofs = csv.reader(fh_proofs, delimiter=',')
	proofs = [row for row in icsv_proofs]
	fh_proofs.close()

	targets = [row[0] for row in proofs]
	proof_bodies = [row[1:] for row in proofs]

	label_params_len = len(label_params.values()[0])

	nninput = []
	nnoutput = []
	expected = [] # the label of the target step in the proof
	for itarget, target in enumerate(targets):
		for ibod, bod in enumerate(proof_bodies[itarget]):
			inputvals = [float(sval) for sval in label_params[target]]
			nnoutput.append([float(sval) for sval in label_params[bod]])
			expected.append(bod)
			for runup in range(ibod-runup_num, ibod):
				if runup < 0:
					inputvals += [0.0 for i in range(label_params_len)]
				else:
					inputvals += [float(sval) for sval in label_params[proof_bodies[itarget][runup]]]
			nninput.append(inputvals)

	combined = list(zip(nninput, nnoutput, expected))
	random.shuffle(combined)
	nninput, nnoutput, expected = zip(*combined)
	return nninput, nnoutput, label_params, expected

def train():
	nninput, nnoutput, _ = init_data()
	input_rec_len = len(nninput[0])
	output_rec_len = len(nnoutput[0])
	numrecs = len(nninput)
	tinput, toutput, logits, tloss = graph_create(input_rec_len, output_rec_len)
	learn(tinput, logits, toutput, nninput, nnoutput, numrecs, tloss)


	return nninput, nnoutput, input_rec_len, output_rec_len, numrecs, tinput, toutput, logits

def runinf():
	nninput, nnoutput, label_params, expected = init_data()
	input_rec_len = len(nninput[0])
	output_rec_len = len(nnoutput[0])
	numrecs = len(nninput)
	labels_by_order = [okey for okey in label_params]
	ordered_tbl = [label_params[okey] for okey in label_params]
	allvals = []
	for row in ordered_tbl:
		sqrtsum = math.sqrt(sum([x ** 2 for x in row]))
		if sqrtsum < 1e-10:
			allvals.append([0.0 for x in row])
		else:
			allvals.append([x / sqrtsum for x in row])
	tinput, toutput, logits, tloss = graph_create(input_rec_len, output_rec_len)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, data_dir+tfchk_fname)
		# returned = []
		num_batches = numrecs / batch_size
		for ibatch in range(num_batches):
			batch_start = ibatch * batch_size
			fd = {	tinput: nninput[batch_start : batch_start+batch_size],
					toutput: nnoutput[batch_start : batch_start+batch_size] }
			errval1, oneoutput = sess.run([tloss, toutput], feed_dict=fd)
			if ibatch is 0: returned = oneoutput
			else:
				returned = np.append(returned, oneoutput, axis=0)
			print 'err: ', errval1

	position_sum = 0
	for iout, testrec in enumerate(returned):
		testrec_sqrt = math.sqrt(sum([x ** 2 for x in testrec]))
		testrec = [x / testrec_sqrt for x in testrec]
		CD_pairs = [(sum([val * row[ival] for ival, val in enumerate(testrec)]), irow) for irow, row in enumerate(allvals)]
		CD_pairs = sorted(CD_pairs, key=lambda tup: tup[0], reverse=True)
		for ipair, pair in enumerate(CD_pairs):
			if labels_by_order[pair[1]] == expected[iout]:
				print 'found ', expected[iout], ' in position ', ipair, ' of the returned results '
				position_sum += ipair
				break

	print 'average poition found: ', float(position_sum) / len(returned)


def graph_create(input_rec_len, output_rec_len):


	tinput = tf.placeholder(tf.float32, shape=(batch_size, input_rec_len))
	toutput = tf.placeholder(tf.float32, shape=(batch_size, output_rec_len))

	logits = inference(tinput, batch_size=batch_size, input_rec_len=input_rec_len, num_hidden1_units=1000,
					   num_hidden2_units=300, output_rec_len=output_rec_len)
	tloss = loss(logits, toutput)
	return tinput, toutput, logits, tloss

def learn(tinput, logits, toutput, nninput, nnoutput, numrecs, tloss):

	train_step = tf.train.AdagradOptimizer(0.05).minimize(tloss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	plotvec = []
	plt.figure(1)
	plt.yscale('log')
	plt.ion()
	for step in range(num_learn_iters+1):
		batch_start = random.randint(0, numrecs - batch_size)
		fd = {	tinput: nninput[batch_start : batch_start+batch_size],
				toutput: nnoutput[batch_start : batch_start+batch_size] }
		if step % (num_learn_iters / 100) is 0:
			errval1 = sess.run([tloss], feed_dict=fd)
			print 'step:', step, ', err: ', errval1
			plotvec.append(errval1)
			# plt.clf()
			plt.plot(plotvec)
			plt.pause(0.05)
		sess.run([train_step], feed_dict=fd)

	saver = tf.train.Saver(tf.all_variables())
	saver.save(sess, data_dir+tfchk_fname)
	sess.close()

	while True: plt.pause(0.5)


# train()
runinf()
print('bye')



