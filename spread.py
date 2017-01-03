import csv
from collections import Counter

ifname = '/devlink/data/metamath/setlabels.txt'
ofname = '/devlink/data/metamath/setexpanded.txt'

fhi = open(ifname, 'rb')
b_fname_included = False
reader = csv.reader(fhi, delimiter=',')

allrows = []
labels = []
fnames = []

data_start_col = 1
if b_fname_included:
	data_start_col = 2

for row in reader:
	labels.append(row[0])
	if b_fname_included:
		fnames.append(row[1])
	vals = [int(d) for d in row[data_start_col:]]
	allrows.append(vals)

fhi.close()

maxval = max([max(row) for row in allrows])
maxlen = max([len(row) for row in allrows])

counters = [Counter(row) for row in allrows]
counter_spread = [[float(crow.get(ival, 0)) for ival in range(maxval)] for crow in counters]

# fullrows = []
# for row in allrows:
fullrows = [counter_spread[irow] + [1.0 if ival < len(row) and i == row[ival] else 0.0 for ival in range(maxlen) for i in range(maxval)] for irow, row in enumerate(allrows) ]

fho = open(ofname, 'wt')
writer = csv.writer(fho, delimiter=',')
for irow, row in enumerate(fullrows):
	data = [str(d) for d in row]
	label_prefix = [str(labels[irow])]
	if b_fname_included:
		label_prefix.append(str(fnames[irow]))
	writer.writerow(label_prefix + data)
fho.close()

print('bye')