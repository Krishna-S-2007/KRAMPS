# Import libs
import pandas as pd

# Import dataset
data = pd.read_csv('Dataset/Dataset_preprocess_7.csv')

data.columns = ['frame.number', 'frame.len', 'frame.time', 'frame.time_epoch', 
                'frame.protocols', 'eth.src', 'eth.dst', 'ip.src', 'ip.dst', 'ip.len', 
                'ip.ttl', 'ip.flags', 'ip.frag_offset', 'ip.proto', 'ip.version', 
                'ip.checksum', 'tcp.srcport', 'tcp.dstport', 'tcp.len', 'tcp.seq', 
                'tcp.ack', 'tcp.flags', 'tcp.flags.syn', 'tcp.flags.ack', 
                'tcp.flags.fin', 'tcp.flags.reset', 'tcp.window_size', 'tcp.checksum', 
                'tcp.stream', 'alert']

data.info()

# Save preproccessed dataset
data.to_csv('Dataset/Dataset_clean_7.csv', index=False)