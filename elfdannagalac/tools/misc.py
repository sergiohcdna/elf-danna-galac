###############################################
#   This is an incredible implementation of   #
#   Class decorators taken from:              #
#       Informit                              #
#       Mark Summerfield: Python Descriptors  #
#       https://www.informit.com/articles/    #
###############################################

import os
import time

class GenericDescriptor:

    def __init__(self,getter,setter):

        self.getter = getter
        self.setter = setter

    def __get__(self,instance,owner=None):

        if instance is None:

            return self

        return self.getter(instance)

    def __set__(self,instance,value):

        return self.setter(instance,value)

#   Check if string is valid
def ValidString(attr_name,empty_allowed=True,options=None):

    def decorator(cls):

        name = "__" + attr_name

        def getter(self):

            return getattr(self,name)

        def setter(self,value):

            msg = '\n\t{0} must be a string'.format(attr_name)

            assert isinstance(value,str),msg

            if not empty_allowed and not value:

                raise ValueError(('{0} may not be empty'.format(attr_name)))

            if options is not None and value not in options:

                msg = ('\n\t{0} with value {1} '.format(attr_name,value) +
                       'is not allowed. Valid options are:\n{0}'.format(options))

                raise ValueError(msg)

            setattr(self,name,value)

        setattr(cls,attr_name,GenericDescriptor(getter,setter))

        return cls

    return decorator

def ValidValue(attr_name,min_val=None,max_val=None):

    def decorator(cls):

        name = "__" + attr_name

        def getter(self):

            return getattr(self,name)

        def setter(self,value):

            msg = '\n\t{0} must be a float'.format(attr_name)

            assert isinstance(value,float),msg

            if min_val is not None and value<min_val:

                msg = ('\n\t{0} with value {1} '.format(attr_name,value) +
                       'is below the minimum value allowed: {0}'.format(min_val))

                raise ValueError(msg)

            if max_val is not None and value>max_val:

                msg = ('\n\t{0} with value {1}'.format(attr_name,value) +
                       'is above the maximum value allowed: {0}'.format(max_val))

                raise ValueError(msg)

            setattr(self,name,value)

        setattr(cls,attr_name,GenericDescriptor(getter,setter))

        return cls

    return decorator

def elapsed_time(start,msg=''):

    e       = int(time.time() - start)
    strtime = '{:02d}:{:02d}:{:02d}'.format(e//3600, (e%3600//60), e%60)
    if len(msg) > 0:
        msg = msg + strtime
    else:
        msg = f'Time to finish whatever process you are doing was: {strtime}'

    print(msg)

    return

def checkDir(thispath):
    '''
    This function allow to check if the path belongs 
    to a directory or any other file-type.
    The functions is useful to check if any output path
    corresponds to a valid directory
    '''
    if os.path.exists(thispath) :
        if not os.path.isdir(thispath) :
            raise ValueError('{0} is not a directory'.format(thispath))
        else :
            print('Specified path is a directory. Nothing to do')
    else :
        msg = ('It seems like the path does not exists.')
        print(msg)
        print('Creating directory')
        os.makedirs(thispath)

    return
