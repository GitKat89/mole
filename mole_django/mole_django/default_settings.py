from .settings import *

DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'NAME': 'mole',
        'CLIENT': {
                'host': 'mongo',
                'port': 27017,
            },
        'HOST' : 'mongodb://mongo:27017/',
        'PORT' : 27017,
    }
}