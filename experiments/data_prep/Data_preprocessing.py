#########################################
# Preprocessing
######################################### 

# Import libs
import pandas as pd

# Import dataset
data = pd.read_csv('Dataset/Dataset_clean_1.csv')

# Convert 'frame.time' column to seperate coulumns
year = pd.to_datetime(data['frame.time']).dt.year
month = pd.to_datetime(data['frame.time']).dt.month
day = pd.to_datetime(data['frame.time']).dt.day
hour = pd.to_datetime(data['frame.time']).dt.hour
minute = pd.to_datetime(data['frame.time']).dt.minute
second = pd.to_datetime(data['frame.time']).dt.second

# Add seperated time columns
data['frame.year'] = year
data['frame.month'] = month
data['frame.day'] = day
data['frame.hour'] = hour
data['frame.minute'] = minute
data['frame.second'] = second

# Drop the 'frame.time' column
data = data.drop('frame.time', axis=1)

# Define function to encode 'format_protocols'
def format_protocols(target):
    formatted_protocols = []
    for elements in target:
        # Encoding protocol 'eth:ethertype:ip:tcp' as 1
        if elements == 'eth:ethertype:ip:tcp':
            formatted_protocols.append(1)
        # Encoding protocol 'eth:ethertype:ip:tcp:http' as 2
        elif elements == 'eth:ethertype:ip:tcp:http':
            formatted_protocols.append(2)
        # Encoding protocol 'eth:ethertype:ip:tcp:http:data-text-lines' as 3
        elif elements == 'eth:ethertype:ip:tcp:http:data-text-lines':
            formatted_protocols.append(3)
        # Encoding protocol 'eth:ethertype:ip:tcp:nbss' as 4
        elif elements == 'eth:ethertype:ip:tcp:nbss':
            formatted_protocols.append(4)
        # Encoding protocol 'eth:ethertype:ip:tcp:data' as 5
        elif elements == 'eth:ethertype:ip:tcp:data':
            formatted_protocols.append(5)
        # Encoding protocol 'eth:ethertype:ip:tcp:imap' as 6
        elif elements == 'eth:ethertype:ip:tcp:imap':
            formatted_protocols.append(6)
        # Encoding protocol 'eth:ethertype:ip:tcp:ssh' as 7
        elif elements == 'eth:ethertype:ip:tcp:ssh':
            formatted_protocols.append(7)
        # Encoding protocol 'eth:ethertype:ip:tcp:tls' as 8
        elif elements == 'eth:ethertype:ip:tcp:tls':
            formatted_protocols.append(8)
        # Encoding protocol 'eth:ethertype:ip:tcp:nbss:smb' as 9
        elif elements == 'eth:ethertype:ip:tcp:nbss:smb':
            formatted_protocols.append(9)
        # Encoding protocol 'eth:ethertype:ip:tcp:tls:x509sat:x509sat' as 10
        elif elements == 'eth:ethertype:ip:tcp:tls:x509sat:x509sat':
            formatted_protocols.append(10)
        # Encoding protocol 'eth:ethertype:ip:tcp:http:data' as 11
        elif elements == 'eth:ethertype:ip:tcp:http:data':
            formatted_protocols.append(11)
        # Encoding protocol 'eth:ethertype:ip:tcp:tls:tls' as 12
        elif elements == 'eth:ethertype:ip:tcp:tls:tls':
            formatted_protocols.append(12)
        # Encoding protocol 'eth:ethertype:ip:tcp:http:xml' as 13
        elif elements == 'eth:ethertype:ip:tcp:http:xml':
            formatted_protocols.append(13)
        # Encoding protocol 'eth:ethertype:ip:tcp:http:media' as 14
        elif elements == 'eth:ethertype:ip:tcp:http:media':
            formatted_protocols.append(14)
        # Encoding protocol 'eth:ethertype:ip:tcp:http:png' as 15
        elif elements == 'eth:ethertype:ip:tcp:http:png':
            formatted_protocols.append(15)
        # Encoding protocol 'eth:ethertype:ip:tcp:http:image-gif' as 16
        elif elements == 'eth:ethertype:ip:tcp:http:image-gif':
            formatted_protocols.append(16)
        # Encoding protocol 'eth:ethertype:ip:tcp:http:urlencoded-form' as 17
        elif elements == 'eth:ethertype:ip:tcp:http:urlencoded-form':
            formatted_protocols.append(17)
        # Encoding protocol 'eth:ethertype:ip:tcp:http:image-jfif' as 18
        elif elements == 'eth:ethertype:ip:tcp:http:image-jfif':
            formatted_protocols.append(18)
        else:
            formatted_protocols.append(19)
    return formatted_protocols

# Encoding Protocols
data['frame.protocols'] = format_protocols(data['frame.protocols'])

data.info()

# Save preproccessed dataset
data.to_csv('Dataset/Dataset_preprocess_1.csv', index=False)