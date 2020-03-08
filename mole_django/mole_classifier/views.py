
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.urls import reverse
from django.core.files.storage import FileSystemStorage

import pymongo
#https://api.mongodb.com/python/current/tutorial.htmös
from pymongo import MongoClient
import gridfs


import json

def save_to_mongodb(request): 
    if request.method == 'POST' and request.FILES['myfile']:
        client = MongoClient('mongodb://localhost:27017/')
        db = client["images"]
        fs = gridfs.GridFS(db)
        myfile = request.FILES['myfile']
        fs.put(myfile)
   
        print("finished fct save_to_mongodb")


#from .forms import UploadFileForm


def index(request):
    return render(request, 'mole_classifier/index.html')


def save_image(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile) #clears file from RAM ?
        #uploaded_file_url = fs.url(filename)
        print("finished upload")

def analyze(request):
    """  action="{% url 'analyze' %}" sends a request with an image. dummy data 'context' is sent back to index.html  as a result"""
    #speichert das Bild ins Dateisystem:
    #save_image(request) #comment out, otherwise bytestream is not saved to mongo
    #speichert in Mongo:
    save_to_mongodb(request)
    # ml modell mit bild füttern
    # ergebnis berechnen und zurückliefern an index.html
    context = {"result": json.dumps([1, 0.3, 0.2, 0.5, 0, 0, 0, 0])}
    #return redirect("/mole", context)
    return render(request, 'mole_classifier/index.html', context)
        

    