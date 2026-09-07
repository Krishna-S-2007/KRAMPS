#########################################
# Cleaning Dataset
######################################### 

# Import libs
import pandas as pd

# Import dataset
data = pd.read_csv('Dataset/attack-simulation-alert.csv')

# Drop columns with many NAN values
data = data.drop(['udp.srcport', 'udp.dstport', 'udp.length',
             'udp.checksum', 'icmp.type', 'icmp.code',
             'icmp.checksum', 'http.request.method', 'http.request.uri',
             'http.request.version', 'http.request.full_uri', 'http.response.code',
             'http.user_agent', 'http.content_length_header', 'http.content_type',
             'http.cookie', 'http.host', 'http.referer', 'http.location', 'http.authorization',
             'http.connection', 'dns.qry.name', 'dns.qry.type',
             'dns.qry.class', 'dns.flags.response', 'dns.flags.recdesired',
             'dns.flags.rcode', 'dns.resp.ttl', 'dns.resp.len',
             'smtp.req.command', 'smtp.data.fragment', 'pop.request.command',
             'pop.response', 'imap.request.command',  'imap.response',
             'ftp.request.command', 'ftp.request.arg', 'ftp.response.code',
             'ftp.response.arg', 'ipv6.src', 'ipv6.dst',
             'ipv6.plen', 'eth.type', 'ip.dsfield'], axis=1)

# Drop rows with NAN values
data = data.dropna()

# Save clean dataset
data.to_csv('Dataset/Dataset_clean.csv', index=False)