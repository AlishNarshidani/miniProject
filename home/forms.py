from django import forms
from .models import Notification
import yfinance as yf

def stock_exists(stock_name):
    info = yf.Ticker(stock_name).info
    return 'symbol' in info and len(info) > 1

class NotificationForm(forms.ModelForm):
    class Meta:
        model = Notification
        fields = ['stock_name', 'target_price','price_direction']
    
    def clean_stock_name(self):
        stock_name = self.cleaned_data.get('stock_name')
        if not stock_exists(stock_name):
            raise forms.ValidationError('This stock does not exist.')
        return stock_name