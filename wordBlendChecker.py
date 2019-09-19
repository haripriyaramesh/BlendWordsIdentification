import nltk
import re
import numpy
import textdistance
import itertools
import seaborn as sns
import matplotlib.pyplot as plt  
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import recall_score, precision_score,accuracy_score, classification_report
from pyjarowinkler import distance
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from spellchecker import SpellChecker

spell = SpellChecker(distance=1)

#### READING FROM FILE INTO LIST  ####

candidateList = []
with open("/Users/haripriyaramesh/Documents/Sem 1/Knowledge Technologies/2019S2-COMP90049_proj1-data/candidates.txt") as f:
    fullCandidateList = f.read().splitlines()
candidateList = []
for word in fullCandidateList:
    if not re.search(r'(.)\1\1', word) and not re.search(r'(.).*\1.*\1.*\1.*\1.*', word):
        candidateList.append(word)
femaleNamesSet = set(line.strip().lower() for line in open("/Users/haripriyaramesh/Documents/Sem 1/Knowledge Technologies/2019S2-COMP90049_proj1-data/female.txt"))
maleNamesSet = set(line.strip().lower() for line in open("/Users/haripriyaramesh/Documents/Sem 1/Knowledge Technologies/2019S2-COMP90049_proj1-data/male.txt"))
allnamesSet = femaleNamesSet.union(maleNamesSet)
resultList = [o for o in candidateList if o not in allnamesSet]
finalCandidateList = []
for word in resultList:
    if spell.correction(word) == word:
        finalCandidateList.append(word)
with open("/Users/haripriyaramesh/Documents/Sem 1/Knowledge Technologies/2019S2-COMP90049_proj1-data/dict.txt") as f:
    dictList = f.read().splitlines()
with open("/Users/haripriyaramesh/Documents/Sem 1/Knowledge Technologies/2019S2-COMP90049_proj1-data/blends.txt") as f:
    blendList = f.read().splitlines()
blendDictList = []
trueblendWordList = []

### CALCULATING AVERAGE DISTANCE ###

for entry in blendList:
    dict = {}
    words = entry.split("\t")
    dict['word'] = words[0]
    trueblendWordList.append(words[0])
    dict['prefix'] = words[1]
    dict['suffix'] = words[2]
    dict['preDist'] = distance.get_jaro_distance(
        dict['word'], dict['prefix'], winkler=True, scaling=0.1)
    dict['sufDist'] = distance.get_jaro_distance(
        dict['word'], dict['suffix'], winkler=True, scaling=0.1)
        
### other distances for analysis ###

    # dict['preDist'] = textdistance.hamming.normalized_similarity(dict['word'], dict['prefix'])
    # dict['sufDist'] = textdistance.hamming.normalized_similarity(dict['word'], dict['suffix'])
    # dict['preDist'] = textdistance.levenshtein.normalized_similarity(dict['word'], dict['prefix'])
    # dict['sufDist'] = textdistance.levenshtein.normalized_similarity(dict['word'], dict['suffix'])
    blendDictList.append(dict)
avgPreDist = 0
avgSufDist = 0
for item in blendDictList:
    avgPreDist = item['preDist'] + avgPreDist/len(blendDictList)
    avgSufDist = item['sufDist'] + avgSufDist/len(blendDictList)
print("avg prefix distance: ", avgPreDist/len(blendDictList))
print("avg suffix distance: ", avgSufDist/len(blendDictList))


### ITERATIVELY EXTRACT DICT WORDS FOR A GIVEN CANDIDATE ###

def dictionaryMatches(word):
    j = 2
    prefixDistCheck = False
    suffixDistCheck = False
    while(j <= len(word)-2):
        prefix = word[0:j]
        suffix = word[j:]
        dictprefixList = []
        dictprefixList = [i for i in dictList if i.startswith(prefix)]
        dictsuffixList = [i for i in dictList if i.endswith(suffix)]
        if(not prefixDistCheck):
            for dict in dictprefixList:
                if(distance.get_jaro_distance(word, dict, winkler=True, scaling=0.1) > avgPreDist):
                    prefixDistCheck = True
                    break
        if(not suffixDistCheck):
            for dict in dictsuffixList:
                if(distance.get_jaro_distance(word, dict, winkler=True, scaling=0.1) > avgSufDist):
                    suffixDistCheck = True
                    break   
        if(prefixDistCheck and suffixDistCheck):
            break
        j = j + 1
    if(prefixDistCheck and suffixDistCheck):
        return "True"
    else:
        return "False"




finalList = []
predictedResult = []
actualResult = []
for word in finalCandidateList:
    dict = {}
    dict['word'] = word
    dict['isBlend'] = dictionaryMatches(word)
    print(word, " done")
    predictedResult.append(dict['isBlend'])
    finalList.append(dict)
    if word in trueblendWordList:
        actualResult.append("True")
    else:
        actualResult.append("False")

        



######## F1 measure ######

Accuracy_Score = accuracy_score(actualResult, predictedResult)
precision, recall, fscore, support = score(
    actualResult, predictedResult, average='binary', pos_label="True")

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
print('Accuracy_Score: {}'.format(Accuracy_Score))

with open('predicted.txt', 'w') as f:
    for item in predictedResult:
        f.write("%s\n" % item)

with open('actual.txt', 'w') as f:
    for item in actualResult:
        f.write("%s\n" % item)

cm = confusion_matrix(actualResult, predictedResult)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax) #annot="True" to annotate cells
target_names = ['Yes Blend', 'Not Blend']
print(classification_report(actualResult, predictedResult, target_names=target_names))
# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['YesBlend', 'NotBlend'])
ax.yaxis.set_ticklabels(['NotBlend', 'YesBlend'])
plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.get_cmap("Blues")):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

numpy.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=["True", "False"], title='Confusion matrix')

### CODE TO SAVE PREDICTION INTO A FILE FOR ANALYSIS ###
'''
# print(len(resultList))
print(len(candidateList))
print(len(resultList))
# blendVals = [o for o in candidateList if o in allnamesSet]
afterName = numpy.setdiff1d(candidateList,resultList, assume_unique=True)
print(len(afterName))
# afterName = list(set(candidateList) - set(resultList)).sort()
# afterSpell = list(set(resultList) - set(finalCandidateList)).sort()
afterSpell = numpy.setdiff1d(resultList,finalCandidateList, assume_unique=True)
with open('afterName.txt', 'w') as f:
    for item in afterName:
        f.write("%s\n" % item)
with open('afterSpell.txt', 'w') as f:
    for item in afterSpell:
        f.write("%s\n" % item)
blendValsLeft = [o for o in afterSpell if o in set(trueblendWordList)]
with open('blendValsLeft.txt', 'w') as f:
    for item in blendValsLeft:
        f.write("%s\n" % item)
blendValsList = [item for item in blendvals if item in set(trueblendWordList)]

falseList = ["False"] * (len(fullCandidateList)-len(finalCandidateList))
blendfalseList = ["False"] * (len(fullCandidateList)-len(finalCandidateList)-56)
blentruelist = ["True"]*56
actualResult.extend(blendfalseList)
actualResult.extend(blentruelist)
predictedResult.extend(falseList)

'''