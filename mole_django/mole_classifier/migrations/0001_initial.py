# Generated by Django 2.2.11 on 2020-03-14 22:18

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='UserInput',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sex', models.TextField()),
                ('age_approx', models.TextField()),
                ('anatom_site_general', models.TextField()),
            ],
        ),
    ]
