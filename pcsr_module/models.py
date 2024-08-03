import copy

from .pcsr import PCSR


models = {}


# def register(name):
#     print(name)
#     def decorator(cls):
#         models[name] = cls
#         return cls

#     return decorator


def make(model_spec, args=None, load_sd=False):
    print(f'{model_spec["name"]=}')
    model_args = model_spec["args"]
    print(f"{model_args=}")
    
    model = PCSR(**model_args)
    model.load_state_dict(model_spec["sd"])
    
    return model
