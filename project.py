from nltk.tag import StanfordNERTagger
import wikipedia
from nltk.corpus import wordnet
from nltk.wsd import lesk
import sys
import nltk

def main(argv):
    # Opening file to read and file to write, also preparing StanfordNERTagger.
    st = StanfordNERTagger('/home/sietse/Desktop/Project Tekstanalyse/Project/stanford-ner-2015-12-09/classifiers/english.all.3class.distsim.crf.ser.gz')
    f = open("{0}/en.tok.off.pos".format(argv[1])).readlines()
    g = open("en.tok.off.pos.ent.{0}.{1}".format(argv[1][-9:-6],argv[1][-5:]), "w")
    context = []
    lines = []
    text = []
    sets = []
    bigrams = []
    lines_updated = []
    NER_tags = ["COU", "CIT", "NAT", "PER", "ORG", "ANI", "SPO", "ENT"]
    # Creating context for lesk function and NER- and POS-tagging words.
    print("[Tagging words...]")
    for line in f:
        line = line.split()
        lines.append(line)
        context.append(line[3])
        text.append((line[3],line[4]))  
    chunk = nltk.ne_chunk(text)
    stanford = st.tag(context)
    # Here the words will be made ready to be written to the output file (NER-tag added to the line).
    print("[Applying tags to lines...]")  
    for line in f:
        lemmas = []
        names = []
        line = line.split()
        for i in stanford:
            if line[4] in "NNPS" or "NNS":
                if line[3] in i:
                    if i[1] == "PERSON":
                        line.append("PER")
                    if i[1] == "ORGANIZATION":
                        line.append("ORG")
                    if i[1] == "LOCATION":
                        line.append("LOCATION")
        if len(wordnet.synsets(line[3], 'n')) == 0:
            hyper = []
        if len(wordnet.synsets(line[3], 'n')) == 1:
            synset = wordnet.synsets(line[3], 'n')[0]
            hyper = synset.hypernym_paths()
        if len(wordnet.synsets(line[3], 'n')) > 1:
            synset = lesk(context,line[3],'n')
            hyper = synset.hypernym_paths()
        for i in hyper:
            for e in i:
                lemmas.append(e.lemmas())
        for i in lemmas:
            for e in i:
                names.append(e.name())
        if "country" in names and line[3][0].isupper():
            line.append("COU")
        elif "government" in names and line[3][0].isupper():
            line.append("COU")
        elif "province" in names and line[3][0].isupper():
            line.append("COU")
        elif "state" in names and line[3][0].isupper():
            line.append("COU")   
        elif "city" in names and line[3][0].isupper():
            line.append("CIT")
        elif "sport" in names:
            line.append("SPO")
        elif "animal" in names:
            line.append("ANI")
        elif "entertainment" in names and line[3][0].isupper():
            line.append("ENT") 
        elif "amusement" in names and line[3][0].isupper():
            line.append("ENT")
        elif "island" in names:
            line.append("NAT") 
        elif "water" in names:
            line.append("NAT")
        elif "mountain" in names:
            line.append("NAT")  
        if "LOCATION" in line and not "COU" in line:
            line.append("CIT")
            line.remove("LOCATION")
            if "LOCATION" in line:
                line.remove("LOCATION")
        if "LOCATION" in line and "COU" in line:
            line.remove("LOCATION")
            if "LOCATION" in line:
                line.remove("LOCATION")
        lines_updated.append(line)
    # Bigrams are put in a list to make urls more precise.
    for i,j in zip(lines_updated,lines_updated[1:]): 
        if 0 <= 5 < len(i) and 0 <= 5 < len(j):
            if i[5] in NER_tags and j[5] in NER_tags:          
                bigrams.append(i[3]+" "+j[3])
    # Words are assigned Wikipedia pages and then the whole is written to the output file.
    print("[Attaching Wikipedia urls and writing output file...]")
    for line in lines_updated:
        try:
            if 0 <= 5 < len(line):
                if bigrams:
                    for i in bigrams:
                        if line[3] in i:
                            page = wikipedia.page(i)
                            break
                        else:
                            page = wikipedia.page(line[3])
                else: 
                    page = wikipedia.page(line[3])
                if len(line) >= 6:
                    g.write("{0} {1} {2} {3} {4} {5} {6}\n".format(line[0], line[1], line[2], line[3], line[4], line[5], page.url))
                else:
                    g.write("{0} {1} {2} {3} {4}\n".format(line[0], line[1], line[2], line[3], line[4]))
            else:
                g.write("{0} {1} {2} {3} {4}\n".format(line[0], line[1], line[2], line[3], line[4]))
        # If the wikipedia module does not know what url to attach, a custom url will be added which (probably) leads to a disambiguation page.
        except wikipedia.exceptions.DisambiguationError:
            for i in bigrams:
                if line[3] in i:
                    page = "https://en.wikipedia.org/wiki/{0}".format(i.replace(" ", "_"))
                    break
                else:
                    page = "https://en.wikipedia.org/wiki/{0}".format(line[3])       
            if len(line) >= 6:
                g.write("{0} {1} {2} {3} {4} {5} {6}\n".format(line[0], line[1], line[2], line[3], line[4], line[5], page))
            else:
                g.write("{0} {1} {2} {3} {4}\n".format(line[0], line[1], line[2], line[3], line[4]))
            continue
        # If another error occurs, the line will be written to the file with or without NER-tag and without url.
        except wikipedia.exceptions.PageError:
            if len(line) >= 5:
                g.write("{0} {1} {2} {3} {4} {5}\n".format(line[0], line[1], line[2], line[3], line[4], line[5]))
            else:
                g.write("{0} {1} {2} {3} {4}\n".format(line[0], line[1], line[2], line[3], line[4]))
            continue
    print("[Processed file...]")

if __name__ == "__main__":
    main(sys.argv)

    
    
   
    


