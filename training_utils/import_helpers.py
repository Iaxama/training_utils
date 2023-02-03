import inspect

def get_input_names(function):
    '''get arguments names from function'''
    return inspect.getfullargspec(function)[0]


def filter_dict(dict_, keys):
    return {k: dict_[k] for k in keys if k in dict_}


def filter_kwargs(kwargs, func):
    return filter_dict(kwargs, get_input_names(func))


def import_module_from_string(attr_string):
    split = attr_string.split('.')
    attr_name = split[-1]
    module_name = '.'.join(split[:-1])
    mod = __import__(module_name, fromlist=[attr_name])
    return getattr(mod, attr_name)
