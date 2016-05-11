__author__ = 'dudulez'
import csv


class Read:
    def __init__(self, filename, delimiter=','):
        self.filename = filename
        self.delimiter = delimiter

    def read_from_file(self):
        try:
            self.opened = csv.reader(
                open(self.filename, 'rb'), delimiter=self.delimiter)
            return self.opened

        except IOError as e:
            print e, 1

    @classmethod
    def load_data(cls, filename_string):
        if not filename_string:
            raise IOError
        elif filename_string.endswith('.tsv'):
            file_ = cls(filename_string, delimiter='\t')
            return file_.read_from_file()
        elif filename_string.endswith('.csv'):
            file_ = cls(filename_string, delimiter=',')
            return file_.read_from_file()


reader1 = Read.load_data('/home/dudulez/sentiment_analysis/other_data/test.csv')
for i in reader1:
    print i
    break

reader2 = Read('/home/dudulez/sentiment_analysis/other_data/test.csv', delimiter=',')

for i in reader2.read_from_file():
    print i
    break


