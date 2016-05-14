try:
    import cPickle as pickle
except:
    import pickle


class Serializer:
    def __init__(self, file_list, object_list):
        self.files = file_list
        self.objects = object_list()

    def serialize(self):
        ser_files = [
            open(file_, 'wb') for file_ in self.files]
        for data, file_ in zip(self.objects, ser_files):
            pickle.dump(data, file_)
            file_.close()

class Deserialize:
    def __init__(self, file_list):
        self.files = file_list

    def deserialize(self):
        des_files = [open(file_, 'rb') for file_ in self.files]
        return [pickle.load(o) for o in des_files]