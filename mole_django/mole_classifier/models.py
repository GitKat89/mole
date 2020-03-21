from django.db import models

from djongo.storage import GridFSStorage
from django.conf import settings
gridfs_storage = GridFSStorage(collection='images')

# saves input and file on clicking on analyze.
class UserInput(models.Model):
    sex = models.TextField()
    age_approx = models.TextField()
    anatom_site_general = models.TextField()
    image_file = models.FileField(storage= gridfs_storage)
   

