import math

file_content = None

with open("./Data/pima-CFS-without-desc.csv", 'r') as f:
	file_content = f.read().split()

for line in file_content:
	if len(line.strip()) == 0:
		file_content.remove(line)

yes_count = 0
no_count = 0
yes_entry = []
no_entry = []

for line in file_content:
	if 'yes' in line:
		yes_count += 1
		yes_entry.append(line)
	if 'no' in line:
		no_count += 1
		no_entry.append(line)
print(f"{yes_count} yes | {no_count} no")

strata_size = math.ceil(len(file_content) / 10)
yes_per_strata = math.ceil(strata_size * ((yes_count) / (yes_count + no_count)))
no_per_strata = strata_size - yes_per_strata

out_list = []
curr_strata = []

for i in range(10):
	yes_left = len(yes_entry)
	no_left = len(no_entry)
	for j in range(min(yes_per_strata, yes_left)):
		curr_strata.append(yes_entry.pop(0))
	for j in range(min(no_per_strata, no_left)):
		curr_strata.append(no_entry.pop(0))
	out_list.append([entry for entry in curr_strata])
	curr_strata = []

for fold in out_list:
	count = len(fold)
	yes = 0
	no = 0  
	for line in fold:
		if 'yes' in line:
			yes += 1
		if 'no' in line:
			no += 1
	print(f"Total {count} | {yes} yes {no} no")

with open("./Data/pima-CFS-folds.csv", "a") as f:
	for i in range(10):
		f.write(f"fold{i+1}\n")
		for line in out_list[i]:
			f.write(line+'\n')
		f.write("\n")
