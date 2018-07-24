BIT_ROTATE_TABLE = [
    0x0000, 0x1000, 0x0100, 0x1100, 0x0010, 0x1010, 0x0110, 0x1110, 
    0x0001, 0x1001, 0x0101, 0x1101, 0x0011, 0x1011, 0x0111, 0x1111, 
    0x2000, 0x3000, 0x2100, 0x3100, 0x2010, 0x3010, 0x2110, 0x3110, 
    0x2001, 0x3001, 0x2101, 0x3101, 0x2011, 0x3011, 0x2111, 0x3111, 
    0x0200, 0x1200, 0x0300, 0x1300, 0x0210, 0x1210, 0x0310, 0x1310, 
    0x0201, 0x1201, 0x0301, 0x1301, 0x0211, 0x1211, 0x0311, 0x1311, 
    0x2200, 0x3200, 0x2300, 0x3300, 0x2210, 0x3210, 0x2310, 0x3310, 
    0x2201, 0x3201, 0x2301, 0x3301, 0x2211, 0x3211, 0x2311, 0x3311, 
    0x0020, 0x1020, 0x0120, 0x1120, 0x0030, 0x1030, 0x0130, 0x1130, 
    0x0021, 0x1021, 0x0121, 0x1121, 0x0031, 0x1031, 0x0131, 0x1131, 
    0x2020, 0x3020, 0x2120, 0x3120, 0x2030, 0x3030, 0x2130, 0x3130, 
    0x2021, 0x3021, 0x2121, 0x3121, 0x2031, 0x3031, 0x2131, 0x3131, 
    0x0220, 0x1220, 0x0320, 0x1320, 0x0230, 0x1230, 0x0330, 0x1330, 
    0x0221, 0x1221, 0x0321, 0x1321, 0x0231, 0x1231, 0x0331, 0x1331, 
    0x2220, 0x3220, 0x2320, 0x3320, 0x2230, 0x3230, 0x2330, 0x3330, 
    0x2221, 0x3221, 0x2321, 0x3321, 0x2231, 0x3231, 0x2331, 0x3331, 
    0x0002, 0x1002, 0x0102, 0x1102, 0x0012, 0x1012, 0x0112, 0x1112, 
    0x0003, 0x1003, 0x0103, 0x1103, 0x0013, 0x1013, 0x0113, 0x1113, 
    0x2002, 0x3002, 0x2102, 0x3102, 0x2012, 0x3012, 0x2112, 0x3112, 
    0x2003, 0x3003, 0x2103, 0x3103, 0x2013, 0x3013, 0x2113, 0x3113, 
    0x0202, 0x1202, 0x0302, 0x1302, 0x0212, 0x1212, 0x0312, 0x1312, 
    0x0203, 0x1203, 0x0303, 0x1303, 0x0213, 0x1213, 0x0313, 0x1313, 
    0x2202, 0x3202, 0x2302, 0x3302, 0x2212, 0x3212, 0x2312, 0x3312, 
    0x2203, 0x3203, 0x2303, 0x3303, 0x2213, 0x3213, 0x2313, 0x3313, 
    0x0022, 0x1022, 0x0122, 0x1122, 0x0032, 0x1032, 0x0132, 0x1132, 
    0x0023, 0x1023, 0x0123, 0x1123, 0x0033, 0x1033, 0x0133, 0x1133, 
    0x2022, 0x3022, 0x2122, 0x3122, 0x2032, 0x3032, 0x2132, 0x3132, 
    0x2023, 0x3023, 0x2123, 0x3123, 0x2033, 0x3033, 0x2133, 0x3133, 
    0x0222, 0x1222, 0x0322, 0x1322, 0x0232, 0x1232, 0x0332, 0x1332, 
    0x0223, 0x1223, 0x0323, 0x1323, 0x0233, 0x1233, 0x0333, 0x1333, 
    0x2222, 0x3222, 0x2322, 0x3322, 0x2232, 0x3232, 0x2332, 0x3332, 
    0x2223, 0x3223, 0x2323, 0x3323, 0x2233, 0x3233, 0x2333, 0x3333
    ]
INDICES_ROTATE = [12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3]
INDICES_MIRROR = [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]

def rotateBoard90(b):
    """b is a 64-bit score4 board consists of 4 layers
    Return: a 90 degree rotated board as follow (looking from above)
    C D E F         0 4 8 C
    8 9 A B   ==>   1 5 9 D
    4 5 6 7         2 6 A E
    0 1 2 3         3 7 B F
    """
    return rotateLayer90(b & 0xFFFF) \
        |  rotateLayer90(b >> 16 & 0xFFFF) << 16 \
        |  rotateLayer90(b >> 32 & 0xFFFF) << 32 \
        |  rotateLayer90(b >> 48 & 0xFFFF) << 48

def mirrorBoard(b):
    """b is a 64-bit score4 board
    Return: a mirrored board as follow (looking from above)
    C D E F         0 1 2 3
    8 9 A B   ==>   4 5 6 7
    4 5 6 7         8 9 A B
    0 1 2 3         C D E F
    """
    return (b & 0x000F000F000F000F) << 12 \
        |  (b & 0x00F000F000F000F0) << 4  \
        |  (b & 0x0F000F000F000F00) >> 4  \
        |  (b & 0xF000F000F000F000) >> 12

def rotateLayer90(x):
    """Rotate x which is a 16-bit bitboard 90 degree"""
    return (BIT_ROTATE_TABLE[x >> 8] << 2) | (BIT_ROTATE_TABLE[x & 0xFF])

def rotatePi90(pi):
    newPi = [pi[INDICES_ROTATE[i]] for i in range(16)]
    return newPi

def mirrorPi(pi):
    newPi = [pi[INDICES_MIRROR[i]] for i in range(16)]
    return newPi