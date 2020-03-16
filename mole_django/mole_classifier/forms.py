from django import forms


class UserInputForm(forms.Form):
    CHOICES_GENDER = (
        ("1","male"),
        ("2","female"),
        ("3","other")
    )
    CHOICES_ANATOM = (
        ("1","head/neck"),
        ("2","anterior torso"),
        ("3","posterior torso"),
        ("4","legs"),
        ("5","arms")
    )
    CHOICES_AGE = (
        ("1","0-10"),
        ("2","11-20"),
        ("3","21-30"),
        ("4","31-40"),
        ("5","41-50"),
        ("6","51-60"),
        ("7","61-70"),
        ("8","71-80"),
        ("9","81-90"),
        ("10",">91"),
    )

    sex = forms.ChoiceField(choices=CHOICES_GENDER)
    anatom_site_general = forms.ChoiceField(choices=CHOICES_ANATOM)
    age_approx = forms.ChoiceField(choices=CHOICES_AGE)


    #anatom_site_general = forms.CharField(max_length=100)
    #age_approx = forms.CharField(max_length=10)

