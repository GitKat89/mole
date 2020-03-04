
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.urls import reverse
from django.core.files.storage import FileSystemStorage
import json

#from .forms import UploadFileForm


def index(request):
    return render(request, 'mole_classifier/index.html')


def save_image(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        #uploaded_file_url = fs.url(filename)
        print("finished upload")

def analyze(request):
    """  action="{% url 'analyze' %}" sends a request with an image. dummy data 'context' is sent back to index.html  as a result"""
    #speichert das Bild:
    save_image(request)
    # ml modell mit bild füttern
    # ergebnis berechnen und zurückliefern an index.html
    context = {"result": json.dumps([1, 0.3, 0.2, 0.5, 0, 0, 0, 0])}
    #return redirect("/mole", context)
    return render(request, 'mole_classifier/index.html', context)
        

    