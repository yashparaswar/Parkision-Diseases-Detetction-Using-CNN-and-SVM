'''Dataset link
https://zenodo.org/record/2867216#.XpuGXsgzaUl
'''

from src.lib.RecognitionLib import *
#print("hello")
def testVoice():
    path = "src/trainedModel.sav" #Définition du chemin du model
    clf = loadModel(path) #Chargement du model

    return(predict(clf, "upload/test.wav"))#Predicition