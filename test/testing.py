from analyzer.train import Training
from filter.prepocessing import Read, Filter



file_name = "semeval_test.tsv"

filter = Filter('stem', [1, 2], )

train = Train(filter, csv_file)

train.save_to_json()

csv_file = Read(file_name, delimiter="\t")

for i in list(csv_file.open_and_read)[:10]:
    print i

analizer = Training(csv_file)
