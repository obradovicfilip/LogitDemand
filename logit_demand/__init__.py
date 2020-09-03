"""THIS DOES NOT WORK"""

try:
    from ... import config
    try:
        female_only = config.female_only
        under_65 = config.under_65
    except:
        pass
except:
    pass
