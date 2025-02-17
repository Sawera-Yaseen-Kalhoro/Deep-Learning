import random

class Colors:
    RESET = '\033[om'

    # Standard colors
    BLACK = '\033[30m'
    RED =  '\ 033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGNETA = '\033[35m' 
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Standard colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED =  '\ 033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGNETA = '\033[95m' 
    BRIHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Standard colors
    BG_BLACK = '\033[40m'
    BG_RED =  '\ 033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGNETA = '\033[45m' 
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

class RandColors:
    RESET = '\033[om'

    # List of available colors

    COLOR_LIST = ['\033[31m', '\033[32m', '\033[33m', '\033[34m', '\033[35m', '\033[36m', '\033[37m',
        '\033[90m', '\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m', '\033[97m',
        '\033[40m', '\033[41m', '\033[42m', '\033[43m', '\033[44m', '\033[45m', '\033[46m', '\033[47m']
    
    def random_color(self):
        return random.choice(self.COLOR_LIST)