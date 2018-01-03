
def getCSVFromArff(fileNameArff, outputName):
    
    with open(fileNameArff, 'r') as fin:
        data = fin.read().splitlines(True)
        
    i = 0
    cols = []
    for line in data:
        if ('@data' in line):
            i+= 1
            break
        else:
            #print line
            i+= 1
            if (line.startswith('@attribute')):
                if('{' in line):
                    cols.append(line[11:line.index('{')-1])
                else:
                    cols.append(line[11:line.index('numeric')-1])
    
    headers = ",".join(cols)
    
    with open(outputName, 'w') as fout:
        fout.write(headers)
        fout.write('\n')
        fout.writelines(data[i:])

file_name = "./PersonalitySwipes_full.arff"
output = "./android_swipes.csv"
getCSVFromArff(file_name, output)
print("successfully converted from arff to csv")
