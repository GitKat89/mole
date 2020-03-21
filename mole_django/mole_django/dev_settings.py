from .settings import *

DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'NAME': 'mole',
        'CLIENT': {
                'host': 'localhost',
                'port': 27017,
            },
        'HOST' : 'mongodb://localhost:27017/',
        'PORT' : 27017,
    }
}