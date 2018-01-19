import random
random.seed(233)

dir ='../dataset/train.txt'
images = []

with open(dir) as trainfile:
    while True:
        line = trainfile.readline()
        if not line:
            break
        images.append(line)
print len(images)


trainset = open('../dataset/splittrain.txt','w')
valset = open('../dataset/splitval.txt','w')
for idx,f in enumerate(images):
    if idx>5271:
        trainset.writelines(f)
    else:
        valset.writelines(f)
trainset.close()
valset.close()

