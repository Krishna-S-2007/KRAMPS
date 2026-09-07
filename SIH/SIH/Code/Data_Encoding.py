# Import libs
import pandas as pd

# Encoding the Ethernet source and Destination Addresses
def format_eth(target):
    formatted_target = []
    for elements in target:
        # Encoding Ethernt Address '00:50:56:c0:00:0a' as 1
        if elements == '00:50:56:c0:00:0a':
            formatted_target.append(1)
        # Encoding Ethernet Address '00:0c:29:8f:ca:0a' as 2
        elif elements == '00:0c:29:8f:ca:0a':
            formatted_target.append(2)
        # Encoding Ethernet Address '00:0c:29:94:f6:85' as 3   
        elif elements == '00:0c:29:94:f6:85':
            formatted_target.append(3)
        else:
            formatted_target.append(4)
    return formatted_target

# Encoding the ip source and Destination Addresses
def format_ip(target):
    formatted_target = []
    for elements in target:
        # Encoding ip Address '10.20.30.1' as 1
        if elements == '10.20.30.1':
            formatted_target.append(1)
        # Encoding ip Address '10.20.30.101' as 2   
        elif elements == '10.20.30.101':
            formatted_target.append(2)
        # Encoding ip Address '10.20.30.103' as 3 
        elif elements == '10.20.30.103':
            formatted_target.append(3)
        else:
            formatted_target.append(4)
    return formatted_target

# Encoding the alert
def format_alert(target):
    formatted_target = []
    for elements in target:
        # Encoding the alert 'benign' as 1
        if elements == 'benign':
            formatted_target.append(1)
        #Encoding the alert other than benign as 2
        else:
            formatted_target.append(2)
    return formatted_target

# Import dataset
data = pd.read_csv('Dataset/Dataset_preprocess_7.csv')

# Encoding the Ethernet and ip source and Destination Addresses
data['eth.src'] = format_eth(data['eth.src'])
data['eth.dst'] = format_eth(data['eth.dst'])
data['ip.src'] = format_ip(data['ip.src'])
data['ip.dst'] = format_ip(data['ip.dst'])

# Converting the flags and checksums from hexadecimal to decimal
data['ip.flags'] = data['ip.flags'].apply(lambda x: int(x, 16))
data['ip.checksum'] = data['ip.checksum'].apply(lambda x: int(x, 16))
data['tcp.flags'] = data['tcp.flags'].apply(lambda x: int(x, 16))
data['tcp.checksum'] = data['tcp.checksum'].apply(lambda x: int(x, 16))

# formatting alert column
data['alert'] = format_alert(data['alert'])

data.info()

# Save clean dataset
data.to_csv('Dataset/Dataset_encoded_7.csv', index=False)



