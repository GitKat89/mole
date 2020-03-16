"""
BASE_URL = 'mongodb://localhost:27017/'

DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'NAME': 'mole',
        'HOST': BASE_URL
    }
}
"""
import pymongo
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client["mole"] #db-name = mole?
collection = db['mole_classifier_userinput'] 
# listing all of the collections in our database:
print(db.list_collection_names())
# get the first document from the collection 'collection'
import pprint
print(" --------1st element in collection: ")
pprint.pprint(collection.find_one())
# filter 1
print(" --------1 single entry:")
pprint.pprint(collection.find_one({"sex": "2"}))
# filter several elements: 
print(" --------all entries in 'collection', filtered:")
for entry in collection.find({"sex": "2"}):
    pprint.pprint(entry)

print(" --------number of entries in 'collection':")
print(collection.count_documents({}))

print(" --------number of entries in 'collection', filtered:")
print(collection.count_documents({"sex": "2"}))

filecollection = db['images.chunks'] 
print(" --------number of entries in 'filecollection':")
print(collection.count_documents({}))