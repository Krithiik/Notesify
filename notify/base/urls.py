from django.contrib import admin
from django.urls import path
from . import views

app_name = 'base'
urlpatterns = [
    path('',views.home,name="home"),
    path('speech_to_text/',views.speech_to_text,name='speech_to_text'),
    path('summarise/',views.summarise,name='summarise'),
    path('keywords/',views.keywords,name='keywords'),
     path('text_generation/',views.text_generation,name='text_generation'),
]