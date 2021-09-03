import os, shutil, logging

if not os.path.isdir('./images'):
    os.mkdir('./images')
    logging.warning('Creating "images" folder - {0}'.format(os.path.abspath('./images')))

if not os.path.isdir('./images/temp/'):
    os.mkdir('./images/temp/')
    logging.warning('Creating "temp" folder - {0}'.format(os.path.abspath('./images/temp/')))
else:
    shutil.rmtree('./images/temp/')
    os.mkdir('./images/temp/')
    logging.warning('Initialise "temp" folder - {0}'.format(os.path.abspath('./images/temp/')))

