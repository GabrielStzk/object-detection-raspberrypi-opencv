from pydrive.drive import GoogleDrive 
from pydrive.auth import GoogleAuth 
   
# For using listdir() 
import os 
   
  
# Below code does the authentication 
# part of the code 
gauth = GoogleAuth() 
  
# Creates local webserver and auto 
# handles authentication. 
gauth.LocalWebserverAuth()        
drive = GoogleDrive(gauth) 

def upload_image(img_path):
    
    f = drive.CreateFile({'title': "detection_image.jpg"}) 
    f.SetContentFile(img_path)
    f.Upload() 
    f = None
