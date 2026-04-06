import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'blockdelay_project.settings')
django.setup()

from django.contrib.auth.models import User

user = User.objects.get(username='admin')
user.set_password('admin123')
user.save()
print("✓ Admin password reset to: admin123")
print(f"✓ Username: admin")
print(f"✓ Try logging in now at: /admin/")
