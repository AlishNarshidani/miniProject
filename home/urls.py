from django.urls import path
from home import views

urlpatterns = [
    path('', views.index, name='home'),
    path('prediction', views.prediction, name='about'),
    path('prediction/linearprediction', views.linearprediction, name='linearprediction'),
    path('fullCandle', views.fullCandle, name='about'),
    path('visualization', views.visualization, name='about'),
    path('latestnews', views.latestnews, name='about'),
    path('watchlist', views.watchlist, name='about'),
    path('about', views.about, name='about'),
    path('contact', views.contact, name='contact')
]