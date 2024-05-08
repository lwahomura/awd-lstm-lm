import io
import pickle
import sys

class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "model":
            renamed_module = "awd-lstm-lm"

        return super(Unpickler, self).find_class(renamed_module, name)