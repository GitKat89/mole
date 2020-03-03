
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse
from django.core.files.storage import FileSystemStorage
#from .forms import UploadFileForm


def index(request):
    return render(request, 'mole_classifier/index.html')

def handle_uploaded_file(f):
    print("entered handle_uploaded_file")
    with open('/Users/deinemudda/Desktop/testupload.txt', 'wb+') as destination:
        for chunk in f.chunks():
           destination.write(chunk)

def upload(request):
    print("UPLOAD DEF")
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        print("finished upload")
        
  
    return HttpResponseRedirect(reverse('index'))
    