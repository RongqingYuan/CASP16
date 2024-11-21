import os


monomer_path = "/home2/s439906/data/CASP16/monomers_EU_merge_v/"
monomer_list = [txt for txt in os.listdir(
    monomer_path) if txt.endswith(".txt")]
# sort the list
monomer_list = sorted(monomer_list)
colab_data = []
EUs = []
for txt in monomer_list:
    count = 0
    header = None
    path = monomer_path + txt
    found = False
    with open(path, "r") as f:
        for line in f:
            if count == 0:
                header = line
            else:
                try:
                    data = line.split()
                    index = data[1]
                    print(index)
                except:
                    continue
                if "TS145" in index:
                    colab_data.append(line)
                    found = True

            count += 1
    if not found:
        EUs.append(txt)
with open("CASP16_monomer_EU_colab.txt", "w") as f:
    f.write(header)
    for line in colab_data:
        f.write(line)
print(EUs)
