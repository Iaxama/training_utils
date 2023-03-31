from functools import singledispatchmethod
import yaml

class Params(dict):

    @singledispatchmethod
    def __init__(self, in_arg):
        raise ValueError(f"unsupported type {type(in_arg)} for {in_arg}")


    @__init__.register(dict)
    def _from_dict(self, in_dict):
        self.update(in_dict)
        
        try:
            self['in_channels'] = len(self['features'])
        except:
            self['in_channels'] = 1
            
        if not 'num_channels' in self.keys():
            self['num_channels'] = 64

        if 'window_features' not in self.keys():
            self['window_features'] = self['deltat']
        if 'increment_features' not in self.keys():
            self['increment_features'] = 1 / self['sampling_rate']

        self['window'] = int(self['deltat'] * self['sampling_rate'])
        self['increment'] = int(self['step'] * self['sampling_rate'])

        if self['features'] is not None:
            self['window_features'] = int(self['window_features'] * self['sampling_rate'])
            self['increment_features'] = int(self['increment_features'] * self['sampling_rate'])
            
            
            self['sample_length'] = int((self['window'] - (self['window_features'] - self['increment_features'])) / self['increment_features'])
        else:
            self['sample_length'] = self['legendre_q']
        
        self['lr'] = self['learning_rate']  # Some functions accept the keyword lr instead of learning rate
        self['dt'] = self['deltat']  # Some functions accept the keyword dt instead of deltat
        self['method'] = self['learning_method']  # Some functions accept the kw method instead of learning_method

    @__init__.register(str)
    def _from_file_name(self, file_name):
        in_dict = yaml.load(open(file_name), yaml.FullLoader)
        self._from_dict(in_dict)
