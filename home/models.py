from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Notification(models.Model):
    #user = models.ForeignKey(User, on_delete=models.CASCADE)
    stock_name = models.CharField(max_length=100)
    target_price = models.DecimalField(max_digits=10, decimal_places=2)
    ABOVE = 'AB'
    BELOW = 'BE'
    PRICE_CHOICES = [
        (ABOVE, 'Above'),
        (BELOW, 'Below'),
    ]
    price_direction = models.CharField(
        max_length=2,
        choices=PRICE_CHOICES,
        default=ABOVE,
    )

    def _str_(self):
        return f'{self.user.username} - {self.stock_name} - {self.target_price}'
