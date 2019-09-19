# BlendWordsIdentification
This is a project to identify blend words in social media text
This program is the python implementation of identifying lexical blends present in the candidate.txt file.

To run:

python wordBlendChecker.py

Files needed:
1. candidate.txt
2. blends.txt
3. dict.txt
4. male.txt
5. female.txt
(4, 5 from: http://www.cs.cmu.edu/Groups/AI/util/areas/nlp/corpora/names/0.html)

External libraries used:
-nltk
-re
-numpy
-textdistance
-itertools
-seaborn
-matplotlib
-sklearn
-spellchecker


Code structure:


- the program starts by reading files into python lists
- the average of distances from the blend file is then computed
- candidates are filtered i.removing string with repeated chars using regex ii. Removing names iii. Removing misspelled words
- iteratively calls the function to generate dict words of matching prefix and suffix
- function computes the Jaro Winkler distance for each pair and classfies.
- f1 measure is calculated and depicted.
