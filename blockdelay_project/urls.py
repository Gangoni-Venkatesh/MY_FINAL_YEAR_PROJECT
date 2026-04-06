from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', TemplateView.as_view(template_name='home.html'), name='home'),
    path('accounts/', include(('accounts.urls', 'accounts'), namespace='accounts')),
    path('datasets/', include(('datasets.urls', 'datasets'), namespace='datasets')),
    path('modelsapp/', include(('modelsapp.urls', 'modelsapp'), namespace='modelsapp')),
    path('predictions/', include(('predictions.urls', 'predictions'), namespace='predictions')),
    path('reports/', include(('reports.urls', 'reports'), namespace='reports')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
