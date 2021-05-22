import glob
import csv
import pandas as pd
import numpy as np

basehref = 'ExtractROIs/sub-*_rest_Zscore.csv'

def masker(array,first,last):
    result = np.where(array<=last,array,0)
    result = np.where(result>=first,array,0)
    result = np.where(result>0,1,0)
    return result.astype(bool)

index = np.arange(1,227)
netmask = {'SS': masker(index,0.1,35),
        'COTC': masker(index,35,49),
        'AUDI': masker(index,49,62),
        'DMN': masker(index,62,119),
        'VIS': masker(index,119,150),
        'FPTC': masker(index,150,175),
        'SAL': masker(index,175,193),
        'SUB': masker(index,193,206),
        'VA': masker(index,206,215),
        'DA': masker(index,215,226)
        }

# get the excluded subjects
with open('exclude.txt') as csvfile:
    reader = csv.reader(csvfile,delimiter='\n')
    exclude = [s[0] for s in reader]

# class for subjects
class Subject:
    def __init__(self, num):
        self.num = num
        self.filename = basehref.replace('*','{}').format(self.num)
        self.timeseries = pd.read_csv(
                self.filename,sep='\t',header=None,
                usecols=range(1,227))
        self.timeseries = np.transpose(self.timeseries)
        self.ROIcorrs = pd.DataFrame(
                (np.corrcoef(self.timeseries)))

# find correlation of ROI
def networkcorrs(ROIcorrs, netmask):
    netlist = list(netmask.keys())
    data = np.zeros(len(netlist)**2).reshape(len(netlist),len(netlist))    
    for ref in range(len(netlist)):
        for comp in range(len(netlist)):
            corr = pd.DataFrame(ROIcorrs.iloc[
                netmask[netlist[ref]],netmask[netlist[comp]]])
            corr = corr.mask(corr == 1)
            data[ref,comp] = corr.mean().mean()
    return pd.DataFrame(data,columns=netlist,index=netlist)


# parse directory for subjects
filelist = glob.glob(basehref)
sublist = [int(s.split('-')[1].split('_')[0]) for s in filelist]

# calculate
results = {}
for subnum in sublist:
    sub = Subject(subnum)
    results[str(sub.num)] = networkcorrs(sub.ROIcorrs, netmask)
