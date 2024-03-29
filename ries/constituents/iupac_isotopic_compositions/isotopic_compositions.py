"""List of isotopic compositions of all elements

The data were transferred by hand from Ref. :cite:`Meija2016`.

All abundances in the given table should be normalized to 1, which is an excellent condition to 
check whether the data were copied successfully.

For Ne, K, Kr, Ru, Sn, Os, and Pt the abundances did not exactly add up to 1 due to numerical 
inaccuracies or unmatched last digits in the table.
In those cases, I adjusted the last digit of the least abundant isotope to fix the normalization.
None of the cases required a change of the last digit by more than two units.
In the python code below, the additions/subtractions are shown explicitly.

At the precision level of a typical `ries` calculation, none of the normalization fixes should 
have an impact on the final results.

At the end of this file, all elements up to Z=118 (maximum proton number listed in the AME 2020
data file, see also `ries/constituents/ame2020_masses/mass_1.mas20`) that have no stable isotopes 
are filled with an empty list of isotopes.
"""

isotopic_compositions = {
    0: {},
    1: {
        1: 0.99984426,
        2: 0.00015574,
    },
    2: {
        3: 0.000001343,
        4: 0.999998657,
    },
    3: {
        6: 0.07589,
        7: 0.92411,
    },
    4: {
        9: 1.0,
    },
    5: {
        10: 0.1982,
        11: 0.8018,
    },
    6: {
        12: 0.988922,
        13: 0.011078,
    },
    7: {
        14: 0.996337,
        15: 0.003663,
    },
    8: {
        16: 0.9976206,
        17: 0.0003790,
        18: 0.0020004,
    },
    9: {
        19: 1.0,
    },
    10: {
        20: 0.904838,
        21: 0.002696 + 0.000001,
        22: 0.092465,
    },
    11: {
        23: 1.0,
    },
    12: {
        24: 0.78951,
        25: 0.10020,
        26: 0.11029,
    },
    13: {
        27: 1.0,
    },
    14: {
        28: 0.9222968,
        29: 0.0468316,
        30: 0.0308716,
    },
    15: {
        31: 1.0,
    },
    16: {
        32: 0.9504074,
        33: 0.0074869,
        34: 0.0419599,
        36: 0.0001458,
    },
    17: {
        35: 0.757647,
        37: 0.242353,
    },
    18: {
        36: 0.0033361,
        38: 0.0006289,
        40: 0.9960350,
    },
    19: {
        39: 0.932581,
        40: 0.0001167 + 0.0000003,
        41: 0.067302,
    },
    20: {
        40: 0.96941,
        42: 0.00647,
        43: 0.00135,
        44: 0.02086,
        46: 0.00004,
        48: 0.00187,
    },
    21: {
        45: 1.0,
    },
    22: {
        46: 0.08249,
        47: 0.07437,
        48: 0.73720,
        49: 0.05409,
        50: 0.05185,
    },
    23: {
        50: 0.002497,
        51: 0.997503,
    },
    24: {
        50: 0.043452,
        52: 0.837895,
        53: 0.095006,
        54: 0.023647,
    },
    25: {
        55: 1.0,
    },
    26: {
        54: 0.058450,
        56: 0.917540,
        57: 0.021191,
        58: 0.002819,
    },
    27: {
        59: 1.0,
    },
    28: {
        58: 0.680769,
        60: 0.262231,
        61: 0.011399,
        62: 0.036345,
        64: 0.009256,
    },
    29: {
        63: 0.69174,
        65: 0.30826,
    },
    30: {
        64: 0.491704,
        66: 0.277306,
        67: 0.040401,
        68: 0.184483,
        70: 0.006106,
    },
    31: {
        69: 0.601079,
        71: 0.398921,
    },
    32: {
        70: 0.20526,
        72: 0.27446,
        73: 0.07760,
        74: 0.36523,
        76: 0.07745,
    },
    33: {
        75: 1.0,
    },
    34: {
        74: 0.00863,
        76: 0.09220,
        77: 0.07594,
        78: 0.23685,
        80: 0.49813,
        82: 0.08825,
    },
    35: {
        79: 0.50686,
        81: 0.49314,
    },
    36: {
        78: 0.0035518 - 0.0000008,
        80: 0.0228560,
        82: 0.115930,
        83: 0.114996,
        84: 0.569877,
        86: 0.172790,
    },
    37: {
        85: 0.721654,
        87: 0.278346,
    },
    38: {
        84: 0.005574,
        86: 0.098566,
        87: 0.070015,
        88: 0.825845,
    },
    39: {
        89: 1.0,
    },
    40: {
        90: 0.51452,
        91: 0.11223,
        92: 0.17146,
        94: 0.17380,
        96: 0.02799,
    },
    41: {
        93: 1.0,
    },
    42: {
        92: 0.14649,
        94: 0.09187,
        95: 0.15873,
        96: 0.16673,
        97: 0.09582,
        98: 0.24292,
        100: 0.09744,
    },
    44: {
        96: 0.055420,
        98: 0.018688 - 0.000001,
        99: 0.127579,
        100: 0.125985,
        101: 0.170600,
        102: 0.315519,
        104: 0.186210,
    },
    45: {
        103: 1.0,
    },
    46: {
        102: 0.0102,
        104: 0.1114,
        105: 0.2233,
        106: 0.2733,
        108: 0.2646,
        110: 0.1172,
    },
    47: {
        107: 0.518392,
        109: 0.481608,
    },
    48: {
        106: 0.01249,
        108: 0.00890,
        110: 0.12485,
        111: 0.12804,
        112: 0.24117,
        113: 0.12225,
        114: 0.28729,
        116: 0.07501,
    },
    49: {
        113: 0.04281,
        115: 0.95719,
    },
    50: {
        112: 0.00973,
        114: 0.00659,
        115: 0.00339 - 0.00002,
        116: 0.14536,
        117: 0.07676,
        118: 0.24223,
        119: 0.08585,
        120: 0.32593,
        122: 0.04629,
        124: 0.05789,
    },
    51: {
        121: 0.57213,
        123: 0.42787,
    },
    52: {
        120: 0.00096,
        122: 0.02603,
        123: 0.00908,
        124: 0.04816,
        125: 0.07139,
        126: 0.18952,
        128: 0.31687,
        130: 0.33799,
    },
    53: {
        127: 1.0,
    },
    54: {
        124: 0.000952,
        126: 0.000890,
        128: 0.019102,
        129: 0.264006,
        130: 0.040710,
        131: 0.212324,
        132: 0.269086,
        134: 0.104357,
        136: 0.088573,
    },
    55: {
        133: 1.0,
    },
    56: {
        130: 0.001058,
        132: 0.001012,
        134: 0.024170,
        135: 0.065920,
        136: 0.078532,
        137: 0.112317,
        138: 0.716991,
    },
    57: {
        138: 0.0008881,
        139: 0.9991119,
    },
    58: {
        136: 0.00186,
        138: 0.00251,
        140: 0.88449,
        142: 0.11114,
    },
    59: {
        141: 1.0,
    },
    60: {
        142: 0.27153,
        143: 0.12173,
        144: 0.23798,
        145: 0.08293,
        146: 0.17189,
        148: 0.05756,
        150: 0.05638,
    },
    62: {
        144: 0.03078,
        147: 0.15004,
        148: 0.11248,
        149: 0.13824,
        150: 0.07365,
        152: 0.26740,
        154: 0.22741,
    },
    63: {
        151: 0.47810,
        153: 0.52190,
    },
    64: {
        152: 0.002029,
        154: 0.021809,
        155: 0.147998,
        156: 0.204664,
        157: 0.156518,
        158: 0.248347,
        160: 0.218635,
    },
    65: {
        141: 1.0,
    },
    66: {
        156: 0.00056,
        158: 0.00095,
        160: 0.02329,
        161: 0.18889,
        162: 0.25475,
        163: 0.24896,
        164: 0.28260,
    },
    67: {
        165: 1.0,
    },
    68: {
        162: 0.001391,
        164: 0.016006,
        166: 0.335014,
        167: 0.228724,
        168: 0.269852,
        170: 0.149013,
    },
    69: {
        169: 1.0,
    },
    70: {
        168: 0.00123,
        170: 0.02982,
        171: 0.14086,
        172: 0.21686,
        173: 0.16103,
        174: 0.32025,
        176: 0.12995,
    },
    71: {
        175: 0.974013,
        176: 0.025987,
    },
    72: {
        174: 0.001620,
        176: 0.052604,
        177: 0.185953,
        178: 0.272811,
        179: 0.136210,
        180: 0.350802,
    },
    73: {
        180: 0.0001201,
        181: 0.9998799,
    },
    74: {
        180: 0.001198,
        182: 0.264985,
        183: 0.143136,
        184: 0.306422,
        186: 0.284259,
    },
    75: {
        185: 0.37398,
        187: 0.62602,
    },
    76: {
        184: 0.000197 + 0.000001,
        186: 0.015859,
        187: 0.019644,
        188: 0.132434,
        189: 0.161466,
        190: 0.262584,
        192: 0.407815,
    },
    77: {
        191: 0.37272,
        193: 0.62728,
    },
    78: {
        190: 0.00012 - 0.00001,
        192: 0.00782,
        194: 0.32864,
        195: 0.33775,
        196: 0.25211,
        198: 0.07357,
    },
    79: {
        169: 1.0,
    },
    80: {
        196: 0.00155,
        198: 0.10038,
        199: 0.16938,
        200: 0.23138,
        201: 0.13170,
        202: 0.29743,
        204: 0.06818,
    },
    81: {
        203: 0.29524,
        205: 0.70476,
    },
    82: {
        204: 0.014245,
        206: 0.241447,
        207: 0.220827,
        208: 0.523481,
    },
    83: {
        209: 1.0,
    },
    90: {
        230: 0.00001138,
        232: 0.99998862,
    },
    91: {
        231: 1.0,
    },
    92: {
        234: 0.0000542,
        235: 0.0072041,
        238: 0.9927417,
    },
}

for Z in range(119):
    if Z not in isotopic_compositions:
        isotopic_compositions[Z] = {}
