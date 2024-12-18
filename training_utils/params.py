from functools import singledispatchmethod
import yaml

class Params(dict):

    @singledispatchmethod
    def __init__(self, in_arg):
        raise ValueError(f"unsupported type {type(in_arg)} for {in_arg}")


    @__init__.register(dict)
    def _from_dict(self, in_dict):
        self.update(in_dict)
        
        if not 'frequency_filter' in self.keys():
            self['frequency_filter'] = True
        self['in_channels'] = len(self['features'])
        if not 'num_channels' in self.keys():
            self['num_channels'] = 64

        if 'window_features' not in self.keys():
            self['window_features'] = self['deltat']
        if 'increment_features' not in self.keys():
            self['increment_features'] = 1 / self['sampling_rate']

        if 'window' not in self.keys():
            self['window'] = int(self['deltat'] * self['sampling_rate'])
        if 'increment' not in self.keys():
            self['increment'] = int(self['step'] * self['sampling_rate'])
        
        if type(self['window_features']) is not int: 
            self['window_features'] = int(self['window_features'] * self['sampling_rate'])
        if type(self['increment_features']) is not int: 
            self['increment_features'] = int(self['increment_features'] * self['sampling_rate'])
        
        if not 'sample_length' in self.keys():
            self['sample_length'] = int((self['window'] - (self['window_features'] - self['increment_features'])) / self['increment_features'])

        if 'learning_rate' in self.keys():
            self['lr'] = self['learning_rate']  # Some functions accept the keyword lr instead of learning rate
        if 'deltat' in self.keys():
            self['dt'] = self['deltat']  # Some functions accept the keyword dt instead of deltat
        if 'learning_method' in self.keys():
            self['method'] = self['learning_method']  # Some functions accept the kw method instead of learning_method
        for x in self:
            try:
                self[x] = eval(self[x])
            except:
                pass
        self.used_keys=[]
        
    @__init__.register(str)
    def _from_file_name(self, file_name):
        in_dict = yaml.load(open(file_name), yaml.FullLoader)
        self._from_dict(in_dict)

    def __getitem__(self, key):
        if hasattr(self, 'used_keys'):
            if not key in self.used_keys:
                self.used_keys.append(key)
        return super().__getitem__(key)

    def save_used_keys_to_yaml(self, file_name):
        yaml.dump({x: self[x] for x in self.used_keys}, open(file_name, 'w'))