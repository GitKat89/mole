
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.urls import reverse
from django.core.files.storage import FileSystemStorage
from .models import UserInput

import pymongo
#https://api.mongodb.com/python/current/tutorial.htmös
from pymongo import MongoClient
import gridfs
import json
from .forms import UserInputForm



def save_to_mongodb(request): 
    if request.method == 'POST' and request.FILES['myfile']:
        client = MongoClient('mongodb://localhost:27017/')
        db = client["images"]
        fs = gridfs.GridFS(db)
        myfile = request.FILES['myfile'] #läd Datei als bytearray
        fs.put(myfile) #speichert bytearray in db
   
        print("finished fct save_to_mongodb")

def index(request):
    form = UserInputForm()
    return render(request, 'mole_classifier/index.html', {'form':form})

def save_user_input(request):
    if request.method == 'POST' and request.FILES['myfile']:
        form = UserInputForm(request.POST)
        if form.is_valid():
            sex = form.cleaned_data['sex']
            age_approx = form.cleaned_data['age_approx']
            anatom_site_general = form.cleaned_data['anatom_site_general']
            userinput = UserInput(age_approx = age_approx, sex = sex, anatom_site_general = anatom_site_general, image_file=request.FILES['myfile'])  
            userinput.save()

def feed_ml():
    # Ergebnis berechnen und zurückliefern an index.html
    # TODO: mit Bild füttern und ML Modell aufrufen
    # Dummy Ergebnis: 
    context = {"result": json.dumps([1, 0.3, 0.2, 0.5, 0, 0, 0, 0])}
    return context

def analyze(request):
    """  action="{% url 'analyze' %}" sends a request with an image. dummy data 'context' is sent back to index.html  as a result"""
    #speichert das Bild sowie die Nutzereingaben in die Datenbank
    save_user_input(request)

    # ML Modell mit Bild füttern
    context = feed_ml()
    return render(request, 'mole_classifier/index.html', context)

def info(request):
    return render(request, 'mole_classifier/info.html')