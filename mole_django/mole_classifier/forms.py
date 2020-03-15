from django import forms


class UserInputForm(forms.Form):
    CHOICES_GENDER = (
        ("1","male"),
        ("2","women"),
        ("3","other")
    )
    anatom_site_general = forms.CharField(max_length=100)
    sex = forms.ChoiceField(choices=CHOICES_GENDER)
    age_approx = forms.CharField(max_length=10)

