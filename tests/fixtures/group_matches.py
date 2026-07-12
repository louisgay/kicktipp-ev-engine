"""Hardcoded group match odds for testing.

Scraped from kicktipp matchdays 1-10 on 2 June 2026.
Used as test fixture data - NOT imported by production code.
"""

# Format: (home, away, P(H), P(D), P(A))
KICKTIPP_GROUP_MATCHES: list[tuple[str, str, float, float, float]] = [
    # -- Group A --
    ("Mexico", "South Africa", 0.899, 0.050, 0.052),
    ("South Korea", "Czech Republic", 0.255, 0.401, 0.344),
    ("Czech Republic", "South Africa", 0.741, 0.170, 0.089),
    ("Mexico", "South Korea", 0.698, 0.210, 0.092),
    ("South Africa", "South Korea", 0.119, 0.239, 0.641),
    ("Czech Republic", "Mexico", 0.141, 0.260, 0.599),
    # -- Group B --
    ("Canada", "Bosnia and Herzegovina", 0.485, 0.214, 0.301),
    ("Qatar", "Switzerland", 0.028, 0.027, 0.945),
    ("Switzerland", "Bosnia and Herzegovina", 0.812, 0.141, 0.047),
    ("Canada", "Qatar", 0.843, 0.115, 0.042),
    ("Bosnia and Herzegovina", "Qatar", 0.807, 0.129, 0.064),
    ("Switzerland", "Canada", 0.731, 0.221, 0.048),
    # -- Group C --
    ("Brazil", "Morocco", 0.864, 0.092, 0.045),
    ("Haiti", "Scotland", 0.020, 0.053, 0.927),
    ("Scotland", "Morocco", 0.231, 0.234, 0.535),
    ("Brazil", "Haiti", 0.987, 0.006, 0.008),
    ("Morocco", "Haiti", 0.915, 0.064, 0.021),
    ("Scotland", "Brazil", 0.026, 0.042, 0.932),
    # -- Group D --
    ("United States", "Paraguay", 0.711, 0.156, 0.133),
    ("Australia", "Turkey", 0.070, 0.145, 0.785),
    ("United States", "Australia", 0.724, 0.190, 0.086),
    ("Turkey", "Paraguay", 0.668, 0.252, 0.081),
    ("Paraguay", "Australia", 0.452, 0.358, 0.190),
    ("Turkey", "United States", 0.472, 0.296, 0.232),
    # -- Group E --
    ("Germany", "Curaçao", 0.991, 0.003, 0.005),
    ("Ivory Coast", "Ecuador", 0.215, 0.427, 0.358),
    ("Germany", "Ivory Coast", 0.941, 0.047, 0.012),
    ("Ecuador", "Curaçao", 0.850, 0.121, 0.029),
    ("Curaçao", "Ivory Coast", 0.048, 0.125, 0.827),
    ("Ecuador", "Germany", 0.025, 0.074, 0.902),
    # -- Group F --
    ("Netherlands", "Japan", 0.789, 0.154, 0.057),
    ("Sweden", "Tunisia", 0.791, 0.145, 0.063),
    ("Netherlands", "Sweden", 0.794, 0.163, 0.043),
    ("Tunisia", "Japan", 0.087, 0.158, 0.755),
    ("Japan", "Sweden", 0.386, 0.391, 0.223),
    ("Tunisia", "Netherlands", 0.023, 0.043, 0.933),
    # -- Group G --
    ("Belgium", "Egypt", 0.872, 0.091, 0.036),
    ("Iran", "New Zealand", 0.470, 0.321, 0.210),
    ("Belgium", "Iran", 0.938, 0.048, 0.014),
    ("New Zealand", "Egypt", 0.131, 0.193, 0.676),
    ("New Zealand", "Belgium", 0.026, 0.039, 0.935),
    ("Egypt", "Iran", 0.626, 0.301, 0.073),
    # -- Group H --
    ("Spain", "Cape Verde", 0.993, 0.003, 0.004),
    ("Saudi Arabia", "Uruguay", 0.042, 0.094, 0.864),
    ("Spain", "Saudi Arabia", 0.987, 0.007, 0.006),
    ("Uruguay", "Cape Verde", 0.901, 0.075, 0.024),
    ("Uruguay", "Spain", 0.019, 0.095, 0.886),
    ("Cape Verde", "Saudi Arabia", 0.096, 0.340, 0.564),
    # -- Group I --
    ("France", "Senegal", 0.972, 0.021, 0.007),
    ("Iraq", "Norway", 0.018, 0.048, 0.934),
    ("France", "Iraq", 0.990, 0.005, 0.005),
    ("Norway", "Senegal", 0.629, 0.275, 0.096),
    ("Norway", "France", 0.031, 0.085, 0.884),
    ("Senegal", "Iraq", 0.795, 0.150, 0.054),
    # -- Group J --
    ("Argentina", "Algeria", 0.957, 0.033, 0.010),
    ("Austria", "Jordan", 0.946, 0.042, 0.012),
    ("Argentina", "Austria", 0.812, 0.129, 0.059),
    ("Jordan", "Algeria", 0.048, 0.236, 0.716),
    ("Algeria", "Austria", 0.064, 0.185, 0.751),
    ("Jordan", "Argentina", 0.013, 0.018, 0.969),
    # -- Group K --
    ("Portugal", "DR Congo", 0.985, 0.010, 0.005),
    ("Uzbekistan", "Colombia", 0.027, 0.071, 0.902),
    ("Portugal", "Uzbekistan", 0.985, 0.008, 0.006),
    ("Colombia", "DR Congo", 0.857, 0.114, 0.028),
    ("DR Congo", "Uzbekistan", 0.359, 0.500, 0.141),
    ("Colombia", "Portugal", 0.044, 0.170, 0.786),
    # -- Group L --
    ("England", "Croatia", 0.685, 0.256, 0.059),
    ("Ghana", "Panama", 0.681, 0.260, 0.059),
    ("England", "Ghana", 0.938, 0.046, 0.015),
    ("Panama", "Croatia", 0.023, 0.039, 0.937),
    ("Croatia", "Ghana", 0.828, 0.128, 0.044),
    ("Panama", "England", 0.014, 0.015, 0.972),
]
