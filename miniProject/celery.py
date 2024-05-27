import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'miniProject.settings')

app = Celery('miniProject')

app.config_from_object('django.conf:settings', namespace='CELERY')

app.autodiscover_tasks()

@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    from home.views import check_stock_prices  # Import here

    sender.add_periodic_task(1.0, check_stock_prices.s(), name='check_stock_prices_every_10_minutes')

app.conf.beat_schedule = {
    'check_stock_prices_every_10_minutes': {
        'task': 'home.views.check_stock_prices',
        'schedule': 1.0,
    },
}