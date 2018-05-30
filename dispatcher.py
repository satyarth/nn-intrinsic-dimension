from os.path import exists
import json
import numpy as np
from random import choice

# From https://stackoverflow.com/questions/45068797/how-to-convert-string-int-json-into-real-int-with-json-loads
class Decoder(json.JSONDecoder):
    def decode(self, s):
        result = super().decode(s)  # result = super(Decoder, self).decode(s) for Python 2.x
        return self._decode(result)

    def _decode(self, o):
        if isinstance(o, str):
            try:
                return int(o)
            except ValueError:
                return o
        elif isinstance(o, dict):
            return {self._decode(k): self._decode(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self._decode(v) for v in o]
        else:
            return o

class Dispatcher:
    def __init__(self, state_file='state.json'):
        self.state_file = state_file
        
        if exists(state_file):
            with open(state_file, 'r') as f:
                j = f.read()
                
            self.state = json.loads(j, cls=Decoder)
            self.state['ongoing'] = {}
            self.dump_state()
            
        else:
            print("No state file, initializing")
            self.init_state()
        
    def init_state(self):
        self.state = {
                'queue': {},
                'done': {},
                'ongoing': {},
                'n_jobs': 0,
                'results': []
            }
        self.dump_state()
        
    def dump_state(self):
        with open(self.state_file, 'w') as f:
                json.dump(self.state, f)
                
    def job_id(self):
        id_ = self.state['n_jobs']
        self.state['n_jobs'] += 1
        self.dump_state()
        return int(id_)
        
    def add_job(self, d, seed=None):
        self.state['queue'][self.job_id()] = {'seed': seed if seed else np.random.randint(0, 420),
                                              'd': d}
        self.dump_state()
        
    def get_job(self):
        try:
            job_id = choice(list(set(self.state['queue'].keys()) - set(self.state['ongoing'].keys())))
            
        except IndexError:
            raise IndexError("No more jobs")
            
        self.state['ongoing'][job_id] = self.state['queue'][job_id]
        self.dump_state()
        return job_id, self.state['queue'][job_id]
        
        
    def finish_job(self, job_id, output):
        try:
            self.state['done'][job_id] = self.state['queue'][job_id]
            self.state['ongoing'].pop(job_id)
            self.state['queue'].pop(job_id)
            self.state['results'].append(output)
            self.dump_state()
            
        except KeyError:
            print("Something's fucky")
        
    
        
