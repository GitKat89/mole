import pymongo
#https://api.mongodb.com/python/current/tutorial.htmös
from pymongo import MongoClient
import gridfs

client = MongoClient('mongodb://localhost:27017/')


db = client["images"]
fs = gridfs.GridFS(db)

#load file as byte array:
f= open("./media/54d90e97bae4780ab00c3737.jpeg","rb")

byte_im = f.read()
#store in mongo:
fs.put(byte_im)


#b = fs.put(fs.get(f), filename="foo", bar="baz")