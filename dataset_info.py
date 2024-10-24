'''


'''
class JTROP:
    def __init__(self) -> None:
        self.num_classes=10
        self.class2label={
            '0':'0',
            '1':'1',
            '2':'2',
            '3':'3',
            '4':'4',
            '8':'5',
            '9':'6',
            '10':'7',
            '11':'8',
            '12':'9',
            '13':'10'
        }
    
class JTROP_binary:
    def __init__(self) -> None:
        self.num_classes=2
        self.class2label={
            '0':'0',
            '1':'1',
            '2':'1',
            '3':'1',
            '4':'1',
            '8':'1',
            '9':'1',
            '10':'1',
            '11':'1',
            '12':'1',
            '13':'1'
        }
def get_dbinfo(dataset_name):
    if dataset_name=='JTROP':
        return JTROP()
    elif dataset_name=='JTROP_binary':
        return JTROP_binary()