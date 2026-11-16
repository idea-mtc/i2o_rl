"""Load preserved original modules without shadowing Isaac or vendor utils."""
import importlib.util,sys

def load_original(directory):
    modules={};saved={k:sys.modules.get(k) for k in ['utils','cqn_utils']}
    try:
        for name in ['utils','cqn_utils','i2o_rl']:
            spec=importlib.util.spec_from_file_location('aligned_original_'+name,directory/(name+'.py'))
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            modules[name]=module
            if name!='i2o_rl':sys.modules[name]=module
    finally:
        for name,old in saved.items():
            if old is None:sys.modules.pop(name,None)
            else:sys.modules[name]=old
    return modules['i2o_rl']
